import os
import json


class MicroledSolver(object):
    CLSNAMES = [
        'microled_a', 'microled_b', 'microled_c', 'microled_d',
        'microled_e', 'microled_f', 'microled_g', 'microled_h',
        'microled_i', 'microled_j'
    ]

    def __init__(self, root='../../data/microled_AD'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            if not os.path.exists(cls_dir):
                continue
                
            # Handle Train (Empty for MicroLED as per observation)
            info['train'][cls_name] = []
            
            # Handle Test
            cls_info = []
            test_dir = f'{cls_dir}/test'
            if os.path.exists(test_dir):
                species = os.listdir(test_dir)
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    specie_dir = f'{test_dir}/{specie}'
                    if not os.path.isdir(specie_dir):
                        continue
                        
                    img_names = os.listdir(specie_dir)
                    img_names.sort()
                    
                    if is_abnormal:
                        mask_dir = f'{cls_dir}/ground_truth/{specie}'
                        if os.path.exists(mask_dir):
                            mask_names = os.listdir(mask_dir)
                            mask_names.sort()
                        else:
                            mask_names = []
                    else:
                        mask_names = None

                    for idx, img_name in enumerate(img_names):
                        mask_path = ''
                        if is_abnormal and mask_names and idx < len(mask_names):
                            mask_path = f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}'
                        
                        info_img = dict(
                            img_path=f'{cls_name}/test/{specie}/{img_name}',
                            mask_path=mask_path,
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
            
            info['test'][cls_name] = cls_info
            
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = MicroledSolver(root='../../data/microled_AD')
    runner.run()
