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
    # {"dataset": "mvtec", "path": data_root_mvtec, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"}, 
    {"dataset": "microled", "path": data_root_microled, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"}, 
    {"dataset": "miniled", "path": data_root_miniled, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},   
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
    checkpoint_path = config["checkpoint"]
    class_name = config.get("class_name", "all")

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    # Paths
    save_dir = f"../../output/VAND-APRIL-GAN-master/{test_dataset}/zero_shot"
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

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
        "--image_size", image_size,
        "--class_name", class_name,
        "--visulize_bool"
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running zero-shot test for dataset={test_dataset}...")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
