import os
import subprocess
import sys

# Configuration
device = "0"
# Path to MVTec dataset
data_root = "../../data/mvtec_anomaly_detection"

n_ctx = 12
vl_reduction = 4
pq_mid_dim = 128

# Using MVTec for both train and test as VisA is not available
train_dataset = "mvtec"
test_dataset = "mvtec" 

# Check if data root exists
if not os.path.exists(data_root):
    print(f"Error: Data root not found at {os.path.abspath(data_root)}")
    sys.exit(1)

shots = [0, 1, 2, 4]

for shot in shots:
    if shot == 0:
        seeds = [10]
    else:
        seeds = [10, 20, 30]
    
    for seed in seeds:
        base_dir = f"{n_ctx}_{vl_reduction}_{pq_mid_dim}_train_on_{train_dataset}_3adapters_batch8"
        
        # Paths
        save_dir = f"./results/{base_dir}"
        model_dir = f"./adaptclip_checkpoints/{base_dir}"
        checkpoint_path = f"{model_dir}/epoch_15.pth"
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}, skipping...")
            continue
            
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Construct the command
        cmd = [
            sys.executable, "test.py",
            "--dataset", test_dataset,
            "--test_data_path", data_root,
            "--seed", str(seed),
            "--k_shots", str(shot),
            "--checkpoint_path", checkpoint_path,
            "--save_path", save_dir,
            "--features_list", "6", "12", "18", "24",
            "--image_size", "518",
            "--batch_size", "8",
            "--n_ctx", str(n_ctx),
            "--vl_reduction", str(vl_reduction),
            "--pq_mid_dim", str(pq_mid_dim),
            "--visual_learner",
            "--textual_learner",
            "--pq_learner",
            "--pq_context"
        ]
        
        # Set environment variable for CUDA
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        
        print(f"Running test for shot={shot}, seed={seed}...")
        # print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            # Optionally continue or break
