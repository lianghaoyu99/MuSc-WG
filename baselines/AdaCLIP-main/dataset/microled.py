import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

MICROLED_CLS_NAMES = [
    'microled_TypeA_1', 'microled_TypeA_2', 'microled_TypeA_3', 'microled_TypeA_4',
    'microled_TypeA_5', 'microled_TypeA_6', 'microled_TypeA_7', 'microled_TypeA_8',
    'microled_TypeA_9', 'microled_TypeA_10', 'microled_TypeA_11', 'microled_TypeA_12',
    'microled_TypeA_13', 'microled_TypeA_14', 'microled_TypeA_15', 'microled_TypeA_16',
    'microled_TypeA_17', 'microled_TypeA_18', 'microled_TypeA_19', 'microled_TypeA_20'
]
MICROLED_ROOT = os.path.join(DATA_ROOT, 'microled_AD')

class MicroLEDDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=MICROLED_CLS_NAMES, aug_rate=0.2, root=MICROLED_ROOT, training=True):
        super(MicroLEDDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )