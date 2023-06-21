
import os
from .cfg import get_cfg_defaults


def get_dataset_cfg(dataset):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_cfg_path = os.path.join(current_dir, f'{dataset}.yaml')
    cfg = get_cfg_defaults()
    cfg.merge_from_file(dataset_cfg_path)
    return cfg