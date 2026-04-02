import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

MINILED_CLS_NAMES = [
    'miniled_TypeA_1', 'miniled_TypeA_2', 'miniled_TypeB_1', 'miniled_TypeB_2',
]
MINILED_ROOT = os.path.join(DATA_ROOT, 'miniled_AD')

class MiniLEDDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=MINILED_CLS_NAMES, aug_rate=0.2, root=MINILED_ROOT, training=True):
        super(MiniLEDDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )