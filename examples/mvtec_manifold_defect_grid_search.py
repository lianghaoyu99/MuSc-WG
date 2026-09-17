"""Resumable Mani grid search for every MVTec product and defect type.

For each product category and hyperparameter pair, MuSc inference is executed
once. The resulting predictions are then evaluated for the whole category and
for every defect separately. A defect-level evaluation contains all normal
(`good`) images and only the selected defect images.

Resume behavior is enabled by default. Progress is committed after one complete
``category x lambda x temperature`` trial, so rerunning the same command skips
all successful trials. If interrupted inside a trial, only that trial is redone.
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

import numpy as np
import torch

sys.path.append(os.getcwd())

from models.musc import MuSc
from utils.load_config import load_yaml
from utils.metrics import compute_metrics


RESULT_FIELDS = [
    "signature",
    "category",
    "defect",
    "lambda_m",
    "temperature",
    "good_count",
    "defect_count",
    "image_auroc",
    "image_f1",
    "image_ap",
    "pixel_auroc",
    "pixel_f1",
    "pixel_ap",
    "aupro",
    "selection_metric",
    "selection_score",
    "ms_per_image",
    "gpu_allocated_mb",
    "gpu_reserved_mb",
]

STATUS_FIELDS = [
    "signature",
    "category",
    "lambda_m",
    "temperature",
    "status",
    "error",
]

COMPARISON_FIELDS = [
    "signature",
    "category",
    "defect",
    "selection_metric",
    "baseline_lambda_m",
    "baseline_selection_score",
    "best_mani_lambda_m",
    "best_mani_temperature",
    "best_mani_selection_score",
    "delta_selection_score",
    "baseline_image_auroc",
    "best_mani_image_auroc",
    "delta_image_auroc",
    "baseline_pixel_auroc",
    "best_mani_pixel_auroc",
    "delta_pixel_auroc",
    "baseline_pixel_ap",
    "best_mani_pixel_ap",
    "delta_pixel_ap",
    "baseline_aupro",
    "best_mani_aupro",
    "delta_aupro",
    "mani_improves_selection",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search Mani lambda/tau for all MVTec categories and report the "
            "best parameters for every defect type."
        )
    )
    parser.add_argument("--config", default="./configs/musc.yaml")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the configured MVTec dataset directory.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional product subset. The default is every configured class.",
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.3, 0.5],
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
            "Metric used to select each defect's best pair. mean_pixel is "
            "the mean of pixel AUROC, pixel AP, and AUPRO."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./output/mvtec_manifold_defect_grid_search",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard result files in output-dir and start from the beginning.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the search plan without loading the model.",
    )
    return parser.parse_args()


def canonical_float(value):
    return format(float(value), ".12g")


def trial_key(signature, category, lambda_m, temperature):
    return (
        signature,
        category,
        canonical_float(lambda_m),
        canonical_float(temperature),
    )


def result_key(row):
    return trial_key(
        row["signature"],
        row["category"],
        row["lambda_m"],
        row["temperature"],
    ) + (row["defect"],)


def trial_pairs(lambda_values, tau_values):
    pairs = []
    for lambda_m in lambda_values:
        if lambda_m < 0:
            raise ValueError("All lambda values must be non-negative.")
        selected_taus = tau_values[:1] if lambda_m == 0 else tau_values
        for temperature in selected_taus:
            if temperature <= 0:
                raise ValueError("All tau values must be positive.")
            pairs.append((float(lambda_m), float(temperature)))
    return pairs


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


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )
    os.replace(temporary, path)


def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def resolve_categories(cfg, data_path, requested):
    if requested:
        categories = requested
    else:
        categories = list(
            cfg["testing"]
            .get("manifold_parameters_by_dataset", {})
            .get("mvtec_ad", {})
            .get("categories", {})
        )
        if not categories:
            categories = sorted(
                path.name for path in Path(data_path).iterdir() if path.is_dir()
            )

    missing = [
        category
        for category in categories
        if not (Path(data_path) / category / "test").is_dir()
    ]
    if missing:
        raise ValueError(f"MVTec classes not found under data path: {missing}")
    return categories


def discover_defects(data_path, category):
    test_dir = Path(data_path) / category / "test"
    return sorted(
        path.name
        for path in test_dir.iterdir()
        if path.is_dir() and path.name != "good"
    )


def make_signature(cfg, data_path, seed, selection_metric):
    signature_data = {
        "dataset": "mvtec_ad",
        "data_path": str(Path(data_path).resolve()),
        "seed": seed,
        "selection_metric": selection_metric,
        "subgroup_policy": "all_good_plus_one_defect",
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


def build_result(
    signature,
    category,
    defect,
    lambda_m,
    temperature,
    good_count,
    defect_count,
    image_metric,
    pixel_metric,
    metric,
    ms_per_image,
    gpu_allocated_mb,
    gpu_reserved_mb,
):
    result = {
        "signature": signature,
        "category": category,
        "defect": defect,
        "lambda_m": float(lambda_m),
        "temperature": float(temperature),
        "good_count": int(good_count),
        "defect_count": int(defect_count),
        "image_auroc": float(image_metric[0]),
        "image_f1": float(image_metric[1]),
        "image_ap": float(image_metric[2]),
        "pixel_auroc": float(pixel_metric[0]),
        "pixel_f1": float(pixel_metric[1]),
        "pixel_ap": float(pixel_metric[2]),
        "aupro": float(pixel_metric[3]),
        "selection_metric": metric,
        "ms_per_image": float(ms_per_image),
        "gpu_allocated_mb": float(gpu_allocated_mb),
        "gpu_reserved_mb": float(gpu_reserved_mb),
    }
    result["selection_score"] = selection_score(result, metric)
    return result


def best_by_defect(rows, signature, metric):
    best = {}
    for row in rows:
        if row.get("signature") != signature:
            continue
        category = row["category"]
        defect = row["defect"]
        score = float(row["selection_score"])
        current = best.setdefault(category, {}).get(defect)
        if current is not None and score <= current["selection_score"]:
            continue
        lambda_m = float(row["lambda_m"])
        best[category][defect] = {
            "lambda_m": lambda_m,
            "temperature": (
                None if lambda_m == 0 else float(row["temperature"])
            ),
            "selection_metric": metric,
            "selection_score": score,
            "image_auroc": float(row["image_auroc"]),
            "pixel_auroc": float(row["pixel_auroc"]),
            "pixel_ap": float(row["pixel_ap"]),
            "aupro": float(row["aupro"]),
        }
    return best


def baseline_vs_best_mani(rows, signature, metric):
    grouped = {}
    for row in rows:
        if row.get("signature") != signature:
            continue
        grouped.setdefault((row["category"], row["defect"]), []).append(row)

    comparisons = []
    for (category, defect), group in sorted(grouped.items()):
        baseline_candidates = [
            row for row in group if float(row["lambda_m"]) == 0
        ]
        mani_candidates = [
            row for row in group if float(row["lambda_m"]) > 0
        ]
        if not baseline_candidates or not mani_candidates:
            continue

        baseline = max(
            baseline_candidates,
            key=lambda row: float(row["selection_score"]),
        )
        best_mani = max(
            mani_candidates,
            key=lambda row: float(row["selection_score"]),
        )
        comparison = {
            "signature": signature,
            "category": category,
            "defect": defect,
            "selection_metric": metric,
            "baseline_lambda_m": 0.0,
            "baseline_selection_score": float(
                baseline["selection_score"]
            ),
            "best_mani_lambda_m": float(best_mani["lambda_m"]),
            "best_mani_temperature": float(best_mani["temperature"]),
            "best_mani_selection_score": float(
                best_mani["selection_score"]
            ),
        }
        comparison["delta_selection_score"] = (
            comparison["best_mani_selection_score"]
            - comparison["baseline_selection_score"]
        )
        for name in ("image_auroc", "pixel_auroc", "pixel_ap", "aupro"):
            baseline_value = float(baseline[name])
            mani_value = float(best_mani[name])
            comparison[f"baseline_{name}"] = baseline_value
            comparison[f"best_mani_{name}"] = mani_value
            comparison[f"delta_{name}"] = mani_value - baseline_value
        comparison["mani_improves_selection"] = (
            comparison["delta_selection_score"] > 0
        )
        comparisons.append(comparison)
    return comparisons


def comparison_as_nested_json(comparisons):
    nested = {}
    for row in comparisons:
        category = row["category"]
        defect = row["defect"]
        nested.setdefault(category, {})[defect] = {
            key: value
            for key, value in row.items()
            if key not in ("signature", "category", "defect")
        }
    return nested


def write_summaries(
    result_rows,
    signature,
    metric,
    best_path,
    comparison_csv_path,
    comparison_json_path,
):
    write_json_atomic(
        best_path, best_by_defect(result_rows, signature, metric)
    )
    comparisons = baseline_vs_best_mani(result_rows, signature, metric)
    write_csv_atomic(
        comparison_csv_path, comparisons, COMPARISON_FIELDS
    )
    write_json_atomic(
        comparison_json_path, comparison_as_nested_json(comparisons)
    )
    return comparisons


def main():
    args = parse_args()
    pairs = trial_pairs(args.lambda_values, args.tau_values)
    cfg = deepcopy(load_yaml(args.config))
    cfg["testing"]["use_manifold"] = True
    data_path = args.data_path or cfg["datasets"].get(
        "data_paths_by_dataset", {}
    ).get("mvtec_ad", "./data/mvtec_anomaly_detection/")
    if not Path(data_path).is_dir():
        raise ValueError(f"MVTec data path does not exist: {data_path}")

    categories = resolve_categories(cfg, data_path, args.classes)
    signature = make_signature(
        cfg, data_path, args.seed, args.selection_metric
    )
    defect_plan = {
        category: discover_defects(data_path, category)
        for category in categories
    }
    total_trials = len(categories) * len(pairs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "defect_grid_results.csv"
    status_path = output_dir / "trial_status.csv"
    best_path = output_dir / "best_by_defect.json"
    comparison_csv_path = output_dir / "baseline_vs_best_mani.csv"
    comparison_json_path = output_dir / "baseline_vs_best_mani.json"
    plan_path = output_dir / "run_plan.json"

    if args.no_resume:
        for path in (
            results_path,
            status_path,
            best_path,
            comparison_csv_path,
            comparison_json_path,
            plan_path,
        ):
            if path.exists():
                path.unlink()

    plan = {
        "signature": signature,
        "data_path": str(Path(data_path).resolve()),
        "categories": defect_plan,
        "lambda_values": [float(value) for value in args.lambda_values],
        "tau_values": [float(value) for value in args.tau_values],
        "trial_pairs": pairs,
        "total_trials": total_trials,
        "selection_metric": args.selection_metric,
        "resume_enabled": not args.no_resume,
        "manifold_layers": cfg["testing"].get("manifold_layers"),
        "manifold_r_list": cfg["testing"].get("manifold_r_list"),
        "subgroup_policy": "all_good_plus_one_defect",
    }
    write_json_atomic(plan_path, plan)

    print(
        f"MVTec classes={len(categories)}, pairs={len(pairs)}, "
        f"trials={total_trials}, signature={signature}",
        flush=True,
    )
    for category, defects in defect_plan.items():
        print(f"  {category}: {', '.join(defects)}")
    if args.dry_run:
        print(f"Plan: {plan_path.resolve()}")
        return

    result_rows = read_csv(results_path)
    status_rows = read_csv(status_path)
    completed = {
        trial_key(
            row["signature"],
            row["category"],
            row["lambda_m"],
            row["temperature"],
        )
        for row in status_rows
        if row.get("status") == "ok"
    }
    result_map = {result_key(row): row for row in result_rows}

    cfg["datasets"]["dataset_name"] = "mvtec_ad"
    cfg["datasets"]["data_path"] = data_path
    cfg["datasets"]["class_name"] = categories[0]
    cfg["testing"]["vis"] = False
    cfg["testing"]["vis_tsne"] = False
    cfg["testing"]["save_excel"] = False
    cfg["testing"]["output_dir"] = str(output_dir / "runtime")
    model = MuSc(cfg, seed=args.seed)
    dataset_parameters = model.manifold_parameters_by_dataset.setdefault(
        "mvtec_ad", {}
    )
    category_parameters = dataset_parameters.setdefault("categories", {})

    finished = 0
    skipped = 0
    for category in categories:
        for lambda_m, temperature in pairs:
            finished += 1
            key = trial_key(signature, category, lambda_m, temperature)
            if key in completed:
                skipped += 1
                print(
                    f"[{finished}/{total_trials}] Resume skip: {category}, "
                    f"lambda={lambda_m}, tau={temperature}",
                    flush=True,
                )
                continue

            print(
                f"\n=== [{finished}/{total_trials}] {category}: "
                f"lambda={lambda_m}, tau={temperature} ===",
                flush=True,
            )
            category_parameters[category] = {
                "lambda_m": lambda_m,
                "temperature": temperature,
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            try:
                (
                    image_metric,
                    pixel_metric,
                    ms_per_image,
                    gpu_allocated_mb,
                    gpu_reserved_mb,
                    predictions,
                ) = model.make_category_data(
                    category, return_predictions=True
                )

                paths = predictions["image_paths"]
                defect_names = np.asarray(
                    [
                        os.path.basename(os.path.dirname(path))
                        for path in paths
                    ]
                )
                good_indices = np.flatnonzero(defect_names == "good")
                trial_results = [
                    build_result(
                        signature,
                        category,
                        "__all__",
                        lambda_m,
                        temperature,
                        len(good_indices),
                        len(paths) - len(good_indices),
                        image_metric,
                        pixel_metric,
                        args.selection_metric,
                        ms_per_image,
                        gpu_allocated_mb,
                        gpu_reserved_mb,
                    )
                ]

                for defect in defect_plan[category]:
                    defect_indices = np.flatnonzero(defect_names == defect)
                    indices = np.concatenate((good_indices, defect_indices))
                    subgroup_image, subgroup_pixel = compute_metrics(
                        predictions["image_labels"][indices],
                        predictions["image_scores"][indices],
                        predictions["pixel_masks"][indices],
                        predictions["pixel_scores"][indices],
                    )
                    defect_result = build_result(
                        signature,
                        category,
                        defect,
                        lambda_m,
                        temperature,
                        len(good_indices),
                        len(defect_indices),
                        subgroup_image,
                        subgroup_pixel,
                        args.selection_metric,
                        ms_per_image,
                        gpu_allocated_mb,
                        gpu_reserved_mb,
                    )
                    trial_results.append(defect_result)
                    print(
                        f"DEFECT_RESULT {category}/{defect}: "
                        f"score={defect_result['selection_score']:.6f}",
                        flush=True,
                    )

                for row in trial_results:
                    result_map[result_key(row)] = row
                result_rows = list(result_map.values())
                write_csv_atomic(results_path, result_rows, RESULT_FIELDS)

                status_rows.append(
                    {
                        "signature": signature,
                        "category": category,
                        "lambda_m": lambda_m,
                        "temperature": temperature,
                        "status": "ok",
                        "error": "",
                    }
                )
                write_csv_atomic(status_path, status_rows, STATUS_FIELDS)
                completed.add(key)
                write_summaries(
                    result_rows,
                    signature,
                    args.selection_metric,
                    best_path,
                    comparison_csv_path,
                    comparison_json_path,
                )
                del predictions
            except KeyboardInterrupt:
                print(
                    "\nInterrupted. Completed trials are saved; rerun the "
                    "same command to resume.",
                    flush=True,
                )
                raise
            except Exception as error:
                status_rows.append(
                    {
                        "signature": signature,
                        "category": category,
                        "lambda_m": lambda_m,
                        "temperature": temperature,
                        "status": "error",
                        "error": repr(error),
                    }
                )
                write_csv_atomic(status_path, status_rows, STATUS_FIELDS)
                print(f"Trial failed: {error!r}", flush=True)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_rows = list(result_map.values())
    comparisons = write_summaries(
        final_rows,
        signature,
        args.selection_metric,
        best_path,
        comparison_csv_path,
        comparison_json_path,
    )
    print(
        f"\nFinished. Skipped {skipped} completed trials.\n"
        f"Results: {results_path.resolve()}\n"
        f"Status: {status_path.resolve()}\n"
        f"Best parameters: {best_path.resolve()}\n"
        f"Baseline comparison: {comparison_csv_path.resolve()}\n"
        f"Compared defects: {len(comparisons)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
