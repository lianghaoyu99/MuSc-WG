import os
import subprocess
import sys

# Add parent directory to path so we can import datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import dataset_classes

# Configuration
device = "0"
# Path to datasets
data_root_mvtec = "../../data/mvtec_anomaly_detection"
data_root_microled = "../../data/microled_AD"
data_root_miniled = "../../data/miniled_AD"

# Output directory (Project root / output / PromptAD-master)
output_dir = "../../output/PromptAD-master"

# Define test configurations
# Note: PromptAD performs few-shot training per class, so we evaluate by running train_cls.py
# which trains the prompts and evaluates on the test set.
# If class_name is "all", it will iterate over all classes in the dataset.
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "transistor"}, # Single class example
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "all"},        # All classes example
    {"dataset": "microled", "path": data_root_microled, "class_name": "all"}, 
    {"dataset": "miniled", "path": data_root_miniled, "class_name": "all"},   # Test all classes in miniled
]

shots = [4]  # Changed to 4-shot. PromptAD requires few-shot normal samples to build feature gallery

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    target_class_config = config["class_name"]

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue
        
    # Determine which classes to run
    if target_class_config.lower() == "all":
        if test_dataset in dataset_classes:
            classes_to_run = dataset_classes[test_dataset]
        else:
            print(f"Error: Dataset {test_dataset} not found in dataset_classes.")
            continue
    else:
        classes_to_run = [target_class_config]

    for target_class in classes_to_run:
        for shot in shots:
            # PromptAD uses seed in a loop for few-shot
            seeds = [111]
            
            for seed in seeds:
                for task_script in ["train_cls.py", "train_seg.py"]:
                    # Construct the command
                    cmd = [
                        sys.executable, task_script,
                        "--dataset", test_dataset,
                        "--data_path", data_root,
                        "--class_name", target_class,
                        "--k-shot", str(shot),
                        "--seed", str(seed),
                        "--gpu-id", str(device),
                        "--root-dir", output_dir, # Add root_dir to specify save path
                        "--vis", "True"  # Add visualization option
                    ]
                    
                    # Set environment variable for CUDA
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = device
                    
                    print(f"\n{'='*50}")
                    print(f"Running {task_script} for dataset={test_dataset}, class={target_class}, shot={shot}, seed={seed}...")
                    print(f"{'='*50}\n")
                    
                    try:
                        subprocess.run(cmd, env=env, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running command for {target_class} ({task_script}): {e}")

