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
    # {"dataset": "mvtec", "path": data_root_mvtec, "data_script": "data/mvtec.py", "checkpoint": "./exps/pretrained/visa_pretrained.pth"}, 
    {"dataset": "microled", "path": data_root_microled, "data_script": "data/microled.py", "checkpoint": "./exps/pretrained/visa_pretrained.pth"}, 
    # {"dataset": "miniled", "path": data_root_miniled, "data_script": "data/miniled.py", "checkpoint": "./exps/pretrained/visa_pretrained.pth"},   
]

# Base arguments
config_path = "./open_clip/model_configs/ViT-L-14-336.json"
model_name = "ViT-L-14-336"
features_list = ["6", "12", "18", "24"]
pretrained = "openai"
image_size = "518"
mode = "zero_shot"

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    data_script = config["data_script"]
    checkpoint_path = config["checkpoint"]

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    # Paths
    save_dir = f"../../output/VAND-APRIL-GAN-master/{test_dataset}/zero_shot"
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # First, run the data preparation script
    print(f"Running data preparation script for {test_dataset}...")
    prep_cmd = [sys.executable, data_script]
    try:
        subprocess.run(prep_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running data preparation script: {e}")
        continue

    # Construct the main test command
    cmd = [
        sys.executable, "test.py",
        "--mode", mode,
        "--dataset", test_dataset,
        "--data_path", data_root,
        "--save_path", save_dir,
        "--config_path", config_path,
        "--checkpoint_path", checkpoint_path,
        "--model", model_name,
        "--features_list", *features_list,
        "--pretrained", pretrained,
        "--image_size", image_size
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running zero-shot test for dataset={test_dataset}...")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
