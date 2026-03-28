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
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "bottle", "checkpoint": "./weights/pretrained_visa_clinicdb.pth"}, 
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "all", "checkpoint": "./weights/pretrained_visa_clinicdb.pth"}, 
    # {"dataset": "microled", "path": data_root_microled, "class_name": "microled_TypeA_1", "checkpoint": "./weights/pretrained_visa_clinicdb.pth"}, 
    {"dataset": "miniled", "path": data_root_miniled, "class_name": "miniled_TypeB_1", "checkpoint": "./weights/pretrained_visa_clinicdb.pth"},   
]

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    target_class = config["class_name"]
    checkpoint_path = config["checkpoint"]

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    # Paths
    if target_class.lower() == "all":
        save_dir = f"../../output/AdaCLIP/{test_dataset}/zero_shot_all"
    else:
        save_dir = f"../../output/AdaCLIP/{test_dataset}/zero_shot_{target_class}"
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the command
    cmd = [
        sys.executable, "test.py",
        "--testing_data", test_dataset,
        "--ckt_path", checkpoint_path,
        "--save_path", save_dir,
        "--obj_name", target_class,
        "--testing_model", "dataset",
        "--save_fig", "True"
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running test for dataset={test_dataset}, class={target_class}...")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
