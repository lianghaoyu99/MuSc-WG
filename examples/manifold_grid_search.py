"""Batch-search manifold lambda and temperature for one or more datasets.

Examples:
    python .\examples\manifold_grid_search.py --datasets miniled_ad
    python .\examples\manifold_grid_search.py --datasets mvtec_ad --classes bottle
    python .\examples\manifold_grid_search.py --datasets miniled_ad microled_ad \
        --lambda-values 0 0.02 0.05 0.1 --tau-values 0.3 0.5 0.8 1.0

The script disables visualization, t-SNE and Excel output, writes every trial
to CSV immediately, and resumes completed trials by default.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.append(os.getcwd())

from models.musc import MuSc
from utils.load_config import load_yaml


RESULT_FIELDS = [
    "signature",
    "dataset",
    "category",
    "lambda_m",
    "temperature",
    "image_auroc",
    "image_f1",
    "image_ap",
    "pixel_auroc",
    "pixel_f1",
    "pixel_ap",
    "aupro",
    "selection_score",
    "selection_metric",
    "ms_per_image",
    "gpu_allocated_mb",
    "gpu_reserved_mb",
    "status",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-search soft-manifold lambda and temperature."
    )
    parser.add_argument("--config", default="./configs/musc.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names, or 'all'. Defaults to datasets.dataset_name.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override data path; valid only when one dataset is selected.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional category subset. Defaults to all configured categories.",
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.1, 0.2],
    )
    parser.add_argument(
        "--tau-values",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.8, 1.0],
    )
    parser.add_argument(
        "--selection-metric",
        choices=[
            "mean_pixel",
            "mean_core",
            "image_auroc",
            "pixel_auroc",
            "pixel_ap",
            "aupro",
        ],
        default="mean_pixel",
        help=(
            "Metric used to select the best pair. mean_pixel averages pixel "
            "AUROC, pixel AP and AUPRO; mean_core also includes image AUROC."
        ),
    )
    parser.add_argument(
        "--output-dir", default="./output/manifold_grid_search"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed rows already present in the result CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned trials without loading the model.",
    )
    return parser.parse_args()


def selection_score(result, metric):
    if metric == "mean_pixel":
        return sum(
            result[name] for name in ("pixel_auroc", "pixel_ap", "aupro")
        ) / 3.0
    if metric == "mean_core":
        return sum(
            result[name]
            for name in ("image_auroc", "pixel_auroc", "pixel_ap", "aupro")
        ) / 4.0
    return result[metric]


def trial_pairs(lambda_values, tau_values):
    """Avoid rerunning equivalent temperatures when lambda is zero."""
    pairs = []
    for lambda_m in lambda_values:
        if lambda_m < 0:
            raise ValueError("All lambda values must be non-negative.")
        selected_taus = tau_values[:1] if lambda_m == 0 else tau_values
        for temperature in selected_taus:
            if temperature <= 0:
                raise ValueError("All temperature values must be positive.")
            pairs.append((float(lambda_m), float(temperature)))
    return pairs


def resolve_datasets(cfg, requested):
    configured = cfg["datasets"].get("data_paths_by_dataset", {})
    if not requested:
        return [cfg["datasets"]["dataset_name"]]
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(configured)
    return requested


def resolve_categories(cfg, dataset, requested):
    if requested:
        return requested
    dataset_parameters = (
        cfg["testing"]
        .get("manifold_parameters_by_dataset", {})
        .get(dataset, {})
    )
    categories = list((dataset_parameters.get("categories", {}) or {}).keys())
    if not categories:
        raise ValueError(
            f"No categories configured for {dataset}; pass --classes explicitly."
        )
    return categories


def run_signature(cfg, dataset, data_path, seed):
    signature_data = {
        "dataset": dataset,
        "data_path": os.path.abspath(data_path),
        "seed": seed,
        "datasets": {
            key: cfg["datasets"].get(key)
            for key in ("img_resize", "divide_num")
        },
        "models": cfg["models"],
        "testing": {
            key: cfg["testing"].get(key)
            for key in (
                "use_rscin",
                "use_manifold",
                "manifold_level",
                "manifold_fusion",
                "manifold_normalize",
                "manifold_layers",
                "manifold_r_list",
                "manifold_grid_size",
                "manifold_ref_chunk_size",
            )
        },
    }
    encoded = json.dumps(signature_data, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def load_existing_rows(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def completed_keys(rows):
    return {
        (
            row["signature"],
            row["dataset"],
            row["category"],
            float(row["lambda_m"]),
            float(row["temperature"]),
        )
        for row in rows
        if row.get("status") == "ok"
    }


def append_result(csv_path, result):
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({name: result.get(name, "") for name in RESULT_FIELDS})


def write_best_summary(rows, signatures, metric, output_path):
    best = {}
    for row in rows:
        dataset = row.get("dataset")
        if row.get("status") != "ok" or row.get("signature") != signatures.get(dataset):
            continue
        category = row["category"]
        score = float(row["selection_score"])
        current = best.setdefault(dataset, {}).get(category)
        if current is None or score > current["selection_score"]:
            lambda_m = float(row["lambda_m"])
            best[dataset][category] = {
                "temperature": (
                    None if lambda_m == 0 else float(row["temperature"])
                ),
                "lambda_m": lambda_m,
                "selection_metric": metric,
                "selection_score": score,
                "image_auroc": float(row["image_auroc"]),
                "pixel_auroc": float(row["pixel_auroc"]),
                "pixel_ap": float(row["pixel_ap"]),
                "aupro": float(row["aupro"]),
            }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(best, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return best


def main():
    args = parse_args()
    base_cfg = load_yaml(args.config)
    datasets = resolve_datasets(base_cfg, args.datasets)
    if args.data_path and len(datasets) != 1:
        raise ValueError("--data-path can only be used with one dataset.")

    pairs = trial_pairs(args.lambda_values, args.tau_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "grid_results.csv"
    best_path = output_dir / "best_parameters.json"
    if args.no_resume:
        if csv_path.exists():
            csv_path.unlink()
        if best_path.exists():
            best_path.unlink()
    existing_rows = [] if args.no_resume else load_existing_rows(csv_path)
    done = completed_keys(existing_rows)
    all_rows = list(existing_rows)
    signatures = {}

    for dataset in datasets:
        cfg = deepcopy(base_cfg)
        # This script searches Mani hyperparameters, so do not silently run an
        # all-baseline grid if use_manifold was disabled in the input config.
        cfg["testing"]["use_manifold"] = True
        path_map = cfg["datasets"].get("data_paths_by_dataset", {})
        data_path = args.data_path or path_map.get(dataset)
        if not data_path:
            raise ValueError(
                f"No data path configured for {dataset}; use --data-path."
            )
        if not os.path.isdir(data_path):
            print(f"Skipping {dataset}: data path does not exist: {data_path}")
            continue

        categories = resolve_categories(cfg, dataset, args.classes)
        signature = run_signature(cfg, dataset, data_path, args.seed)
        signatures[dataset] = signature
        planned = len(categories) * len(pairs)
        print(
            f"Dataset={dataset}, categories={len(categories)}, "
            f"trials={planned}, signature={signature}, "
            f"manifold_layers={cfg['testing'].get('manifold_layers')}, "
            f"manifold_r_list={cfg['testing'].get('manifold_r_list')}"
        )
        if args.dry_run:
            for category in categories:
                print(category, pairs)
            continue

        cfg["datasets"]["dataset_name"] = dataset
        cfg["datasets"]["data_path"] = data_path
        cfg["datasets"]["class_name"] = categories[0]
        cfg["testing"]["vis"] = False
        cfg["testing"]["vis_tsne"] = False
        cfg["testing"]["save_excel"] = False
        cfg["testing"]["output_dir"] = str(output_dir / "runtime")
        model = MuSc(cfg, seed=args.seed)

        dataset_parameters = model.manifold_parameters_by_dataset.setdefault(
            dataset, {}
        )
        category_parameters = dataset_parameters.setdefault("categories", {})

        for category in categories:
            for lambda_m, temperature in pairs:
                key = (signature, dataset, category, lambda_m, temperature)
                if key in done:
                    print(
                        f"Skipping completed: {dataset}/{category} "
                        f"lambda={lambda_m}, tau={temperature}"
                    )
                    continue

                print(
                    f"\n=== {dataset}/{category}: "
                    f"lambda={lambda_m}, tau={temperature} ===",
                    flush=True,
                )
                category_parameters[category] = {
                    "temperature": temperature,
                    "lambda_m": lambda_m,
                }
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                result = {
                    "signature": signature,
                    "dataset": dataset,
                    "category": category,
                    "lambda_m": lambda_m,
                    "temperature": temperature,
                    "selection_metric": args.selection_metric,
                    "status": "ok",
                    "error": "",
                }
                try:
                    image_metric, pixel_metric, avg_time, mem_alloc, mem_reserved = (
                        model.make_category_data(category)
                    )
                    result.update(
                        {
                            "image_auroc": float(image_metric[0]),
                            "image_f1": float(image_metric[1]),
                            "image_ap": float(image_metric[2]),
                            "pixel_auroc": float(pixel_metric[0]),
                            "pixel_f1": float(pixel_metric[1]),
                            "pixel_ap": float(pixel_metric[2]),
                            "aupro": float(pixel_metric[3]),
                            "ms_per_image": float(avg_time),
                            "gpu_allocated_mb": float(mem_alloc),
                            "gpu_reserved_mb": float(mem_reserved),
                        }
                    )
                    result["selection_score"] = selection_score(
                        result, args.selection_metric
                    )
                    done.add(key)
                except Exception as error:
                    result["status"] = "error"
                    result["error"] = repr(error)
                    result["selection_score"] = ""
                    print(f"Trial failed: {error!r}", flush=True)

                append_result(csv_path, result)
                all_rows.append({key: str(value) for key, value in result.items()})
                best = write_best_summary(
                    all_rows,
                    signatures,
                    args.selection_metric,
                    best_path,
                )
                if result["status"] == "ok":
                    print(
                        "GRID_RESULT " + json.dumps(result, sort_keys=True),
                        flush=True,
                    )
                    print(
                        "CURRENT_BEST "
                        + json.dumps(
                            best.get(dataset, {}).get(category, {}),
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.dry_run:
        return
    final_best = write_best_summary(
        all_rows, signatures, args.selection_metric, best_path
    )
    print(f"\nResults: {csv_path.resolve()}")
    print(f"Best parameters: {best_path.resolve()}")
    print(json.dumps(final_best, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
