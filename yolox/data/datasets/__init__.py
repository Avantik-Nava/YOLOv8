from .voc import VOCDetection, AnnotationTransform
from .voc_classes import VOC_CLASSES
from .coco import COCODataset
from .coco_classes import COCO_CLASSES

__all__ = ["VOCDetection", "AnnotationTransform", "VOC_CLASSES",
           "COCODataset", "COCO_CLASSES"]
