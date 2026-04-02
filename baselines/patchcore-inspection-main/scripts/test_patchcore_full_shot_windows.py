import os
import subprocess
import sys

# Configuration
device = "0"
# Path to datasets
data_root_mvtec = "../../data/mvtec_anomaly_detection"
data_root_microled = "../../data/microled_AD"
data_root_miniled = "../../data/miniled_AD"

# Output directory (Project root / output / patchcore)
output_dir = "../../output/PatchCore"

# Define test configurations
# Note: PatchCore natively uses the "run_patchcore.py" script which heavily relies on the "click" library.
# We map the datasets as "mvtec" to reuse the MVTecDataset class structure for microled/miniled as well.
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "transistor"}, 
    {"dataset": "microled", "path": data_root_microled, "class_name": "all"}, 
    {"dataset": "miniled", "path": data_root_miniled, "class_name": "all"},   
]

# We map all to "mvtec" format in patchcore, but pass the different paths.
# PatchCore takes multiple commands chained together in click.
# The general structure is: python bin/run_patchcore.py <save_path> [options] patch_core [options] sampler [options] dataset [options]

backbone_name = "wideresnet50" # PatchCore default backbone

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    target_class = config["class_name"]

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    # Setup the paths
    save_dir = os.path.join(output_dir, test_dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the command using click chaining syntax for patchcore
    cmd = [
        sys.executable, "bin/run_patchcore.py",
        "--gpu", str(device),
        "--seed", "0",
        "--save_patchcore_model",
        "--save_segmentation_images",
        save_dir,
        
        "patch_core",
        "-b", backbone_name,
        "-le", "layer2", "-le", "layer3",  # default patchcore layers for wide_resnet50
        # "--faiss_on_gpu",
        "--pretrain_embed_dimension", "1024",
        "--target_embed_dimension", "1024",
        "--anomaly_scorer_num_nn", "1",
        "--patchsize", "3",
        
        "sampler",
        "-p", "0.1", # default coreset sampling ratio (10%)
        "approx_greedy_coreset",
        
        "dataset",
        "--resize", "256",
        "--imagesize", "224"
    ]
    
    # Subdatasets argument
    if target_class.lower() == "all":
        classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        for c in classes:
            cmd.extend(["-d", c])
    else:
        cmd.extend(["-d", target_class])

    cmd.extend([
        "mvtec", # Dataset type format to use (we reuse mvtec format for miniled/microled)
        data_root
    ])
    
    # Set environment variable to ensure correct working directory for imports
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')) + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"\n{'='*50}")
    print(f"Running PatchCore for dataset={test_dataset}, class={target_class}...")
    print(f"{'='*50}\n")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command for {target_class}: {e}")
