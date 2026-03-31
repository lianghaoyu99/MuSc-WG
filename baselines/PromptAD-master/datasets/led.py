import glob
import os
import random

microled_classes = [
    'microled_TypeA_1', 'microled_TypeA_2', 'microled_TypeA_3', 'microled_TypeA_4',
    'microled_TypeA_5', 'microled_TypeA_6', 'microled_TypeA_7', 'microled_TypeA_8',
    'microled_TypeA_9', 'microled_TypeA_10', 'microled_TypeA_11', 'microled_TypeA_12',
    'microled_TypeA_13', 'microled_TypeA_14', 'microled_TypeA_15', 'microled_TypeA_16',
    'microled_TypeA_17', 'microled_TypeA_18', 'microled_TypeA_19', 'microled_TypeA_20'
]

miniled_classes = [
    'miniled_TypeA_1', 'miniled_TypeA_2', 'miniled_TypeA_3', 'miniled_TypeB_1', 'miniled_TypeC_1'
]

def load_led(category, k_shot, data_path):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if not os.path.exists(root_path):
            return img_tot_paths, gt_tot_paths, tot_labels, tot_types

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                # In microled/miniled, mask name is exactly the same as image name
                gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s)) for s in img_paths]
                
                # Check if GT exists, fallback to blank or ignore? We assume it exists.
                valid_img_paths = []
                valid_gt_paths = []
                for img_p, gt_p in zip(img_paths, gt_paths):
                    if os.path.exists(gt_p):
                        valid_img_paths.append(img_p)
                        valid_gt_paths.append(gt_p)
                    else:
                        valid_img_paths.append(img_p)
                        valid_gt_paths.append(0) # treat as no mask if missing? or error. usually they exist.

                valid_img_paths.sort()
                valid_gt_paths.sort() if all(isinstance(x, str) for x in valid_gt_paths) else None
                img_tot_paths.extend(valid_img_paths)
                gt_tot_paths.extend(valid_gt_paths)
                tot_labels.extend([1] * len(valid_img_paths))
                tot_types.extend([defect_type] * len(valid_img_paths))

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    test_img_path = os.path.join(data_path, category, 'test')
    train_img_path = os.path.join(data_path, category, 'train')
    ground_truth_path = os.path.join(data_path, category, 'ground_truth')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = load_phase(train_img_path, ground_truth_path)
    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = load_phase(test_img_path, ground_truth_path)

    # For few-shot PromptAD, if k_shot > 0, we need to select k_shot normal samples
    if k_shot > 0:
        # Check if there's a seed file like in mvtec, if not, fallback to random sampling with fixed seed
        seed_file = os.path.join('./datasets/seeds_' + ('microled' if category in microled_classes else 'miniled'), category, 'selected_samples_per_run.txt')
        if os.path.exists(seed_file):
            with open(seed_file, 'r') as f:
                files = f.readlines()
            begin_str = f'#{k_shot}: '
            training_indx = []
            for line in files:
                if line.count(begin_str) > 0:
                    strip_line = line[len(begin_str):-1]
                    index = strip_line.split(' ')
                    training_indx = [int(item) for item in index if item]
        else:
            # Fallback: select k_shot random samples from training set
            random.seed(10)
            training_indx = random.sample(range(len(train_img_tot_paths)), min(k_shot, len(train_img_tot_paths)))

        selected_train_img_tot_paths = [train_img_tot_paths[k] for k in training_indx]
        selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in training_indx]
        selected_train_tot_labels = [train_tot_labels[k] for k in training_indx]
        selected_train_tot_types = [train_tot_types[k] for k in training_indx]
    else:
        # k_shot == 0 or -1 means use all train data or none. If 0, empty list.
        if k_shot == 0:
            selected_train_img_tot_paths = []
            selected_train_gt_tot_paths = []
            selected_train_tot_labels = []
            selected_train_tot_types = []
        else:
            selected_train_img_tot_paths = train_img_tot_paths
            selected_train_gt_tot_paths = train_gt_tot_paths
            selected_train_tot_labels = train_tot_labels
            selected_train_tot_types = train_tot_types

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
