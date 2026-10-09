#!/usr/bin/env python3
"""Dataset wrappers (YOLOX yolox/data/datasets/datasets_wrapper.py + mosaicdetection.py port)."""
from torch.utils.data import ConcatDataset, Dataset


class ConcatDatasetWrapper(ConcatDataset):
    pass


class MosaicDetection(Dataset):
    """Wraps a plain detection dataset with mosaic/mixup like YOLOX.

    Our YOLODataset/VOCDetection/COCODataset already do mosaic internally,
    so this is a thin compat wrapper: enable_mosaic toggles dataset.mosaic_prob.
    """

    def __init__(self, dataset, img_size=(640, 640), mosaic=True, preproc=None,
                 degrees=0.0, translate=0.1, shear=0.0, mosaic_scale=(0.1, 2)):
        self.dataset = dataset
        self.img_size = img_size
        self.mosaic = mosaic
        if hasattr(dataset, "mosaic_prob"):
            dataset.mosaic_prob = 1.0 if mosaic else 0.0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def enable_mosaic(self, flag=True):
        self.mosaic = flag
        if hasattr(self.dataset, "mosaic_prob"):
            self.dataset.mosaic_prob = 1.0 if flag else 0.0
