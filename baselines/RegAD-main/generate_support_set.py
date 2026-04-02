import os
import random
import argparse
import torch
from torchvision import transforms
from PIL import Image

def generate_support_set(data_path, class_name, shot, inferences, save_dir):
    """
    Generate fixed support set for RegAD.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    img_dir_train = os.path.join(data_path, class_name, 'train', 'good')
    img_num = sorted(os.listdir(img_dir_train))

    data_train = []
    for img_one in img_num:
        img_dir_one = os.path.join(img_dir_train, img_one)
        data_train.append(img_dir_one)
        
    transform_x = transforms.Compose([
        transforms.Resize((224, 224), getattr(Image, 'Resampling', Image).LANCZOS),
        transforms.ToTensor(),
    ])

    fixed_fewshot_list = []
    
    # Generate deterministic support sets for each inference round
    for inference_round in range(inferences):
        # We use a deterministic seed combined with inference round to get different but reproducible sets
        torch.manual_seed(10 + inference_round)
        perm_indices = torch.randperm(len(data_train))
        indices = perm_indices[:min(shot, len(data_train))]
        
        support_sub_img = None
        for i in indices:
            image_path = data_train[i]
            image = Image.open(image_path).convert('RGB')
            image = transform_x(image)
            image = image.unsqueeze(dim=0)
            if support_sub_img is None:
                support_sub_img = image
            else:
                support_sub_img = torch.cat([support_sub_img, image], dim=0)
        
        fixed_fewshot_list.append(support_sub_img)
        
    save_path = os.path.join(save_dir, f'{shot}_{inferences}.pt')
    torch.save(fixed_fewshot_list, save_path)
    print(f"Saved support set to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate support set for RegAD')
    parser.add_argument('--obj', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--inferences', type=int, default=10)
    args = parser.parse_args()
    
    save_dir = os.path.join('./support_set', args.obj)
    generate_support_set(args.data_path, args.obj, args.shot, args.inferences, save_dir)
