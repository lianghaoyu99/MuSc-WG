"""MiniLED dataset solver for anomaly detection."""

import json
import os

class MiniledSolver:
    CLSNAMES = [
        'miniled_TypeA_1', 'miniled_TypeA_2', 'miniled_TypeB_1', 'miniled_TypeB_2',
    ]

    def __init__(self, root='../../data/miniled_AD'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            if not os.path.exists(cls_dir):
                continue
            
            # Initialize empty lists for both phases
            info['train'][cls_name] = []
            info['test'][cls_name] = []
            
            for phase in ['train', 'test']:
                phase_dir = f'{cls_dir}/{phase}'
                if not os.path.exists(phase_dir):
                    continue
                    
                cls_info = []
                species = os.listdir(phase_dir)
                for specie in species:
                    specie_dir = f'{phase_dir}/{specie}'
                    if not os.path.isdir(specie_dir):
                        continue
                        
                    is_abnormal = True if specie not in ['good'] else False
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
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=mask_path,
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples = anomaly_samples + 1
                            else:
                                normal_samples = normal_samples + 1
                                
                info[phase][cls_name] = cls_info

                
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    runner = MiniledSolver(root='../../data/miniled_AD')
    runner.run()
