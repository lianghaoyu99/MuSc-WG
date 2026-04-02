import os
import subprocess
import sys

# Configuration
device = "0"
# Path to MVTec dataset
data_root_mvtec = "../../data/mvtec_anomaly_detection"
data_root_microled = "../../data/microled_AD"
data_root_miniled = "../../data/miniled_AD"

n_ctx = 12
vl_reduction = 4
pq_mid_dim = 128

# Using visa for both train and test as VisA is default pretrain
train_dataset = "visa"

shots = [4]

# Define test configurations
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "transistor", "train_dataset": "visa"}, # Example
    {"dataset": "microled", "path": data_root_microled, "class_name": "all", "train_dataset": "visa"}, # Test all classes in microled
    {"dataset": "miniled", "path": data_root_miniled, "class_name": "all", "train_dataset": "visa"},   # Test all classes in miniled
]

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    target_class = config["class_name"]
    train_dataset = config["train_dataset"]

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    for shot in shots:
        seeds = [10] # Fixed single seed for 4-shot
        
        for seed in seeds:
            base_dir = f"{n_ctx}_{vl_reduction}_{pq_mid_dim}_train_on_{train_dataset}_3adapters_batch8"
            
            # Paths
            save_dir = f"../../output/AdaptCLIP-main/{test_dataset}/few_shot_{shot}_{target_class}/{base_dir}_seed{seed}"
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
                "--pq_context",
                "--visulize_bool",
                "--class_name", target_class
            ]

            # Set environment variable for CUDA
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            
            print(f"\n{'='*50}")
            print(f"Running Few-Shot test for dataset={test_dataset}, class={target_class}, shot={shot}, seed={seed}...")
            print(f"{'='*50}\n")
            
            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
