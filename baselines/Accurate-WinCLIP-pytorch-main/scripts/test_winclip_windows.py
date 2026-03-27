import os
import subprocess
import sys

# Configuration
device = "0"
# Path to datasets
data_root_mvtec = "../../data/mvtec_anomaly_detection"
data_root_microled = "../../data/microled_AD"
data_root_miniled = "../../data/miniled_AD"

# Define test configurations
# Uncomment the configuration you want to run
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "all"}, 
    # {"dataset": "microled", "path": data_root_microled, "class_name": "all"}, 
    {"dataset": "miniled", "path": data_root_miniled, "class_name": "miniled_TypeA_1"},   
]

few_shot = 0
base_dir_prefix = "winclip_"
model_name = "ViT-B-16-plus-240"
pretrained_weights = "openai"
image_size = 240

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    target_class = config.get("class_name", "all")

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    # Paths
    base_dir = f"{base_dir_prefix}{test_dataset}"
    if target_class.lower() == "all":
        save_dir = f"../../output/WinCLIP/{test_dataset}/zero_shot_all"
    else:
        save_dir = f"../../output/WinCLIP/{test_dataset}/zero_shot_{target_class}"
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the command
    cmd = [
        sys.executable, "reproduce_WinCLIP.py",
        "--dataset", test_dataset,
        "--data_path", data_root,
        "--save_path", save_dir,
        "--model", model_name,
        "--pretrained", pretrained_weights,
        "--k_shot", str(few_shot),
        "--image_size", str(image_size),
        "--class_name", target_class
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running test for dataset={test_dataset}, class={target_class}...")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
