# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py

## shape_sample and its query used for sampling prompts were removed

import copy
# import logging

import cv2
import numpy as np
import torch
from PIL import Image
# from detectron2.config import configurable

from torchvision import transforms
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import MetadataCatalog, Metadata

# from utils import prompt_engineering
# from model.utils import configurable##, PASCAL_CLASSES
# from registration.register_pascalvoc_eval import PASCAL_CLASSES
# from ..visual_sampler import build_shape_sampler
# TODO:  a background should be added to pascal_classes based on SEEM codes for segmentation!!
__all__ = ["PascalVOCSegDatasetMapperIX"]

#from model.utils.config import configurable
PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# This is specifically designed for the COCO dataset.
class PascalVOCSegDatasetMapperIX:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    ##@configurable
    def __init__(
            self,
            is_train=True,
            dataset_name='',
            min_size_test=None,
            max_size_test=None,
            grounding=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test

        t = []
        t.append(transforms.Resize(self.min_size_test, interpolation=Image.BICUBIC, max_size=max_size_test, antialias=True))
        t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
        self.transform = transforms.Compose(t)
        self.ignore_id = 220

        if grounding:
            def _setattr(self, name, value):
                object.__setattr__(self, name, value)

            Metadata.__setattr__ = _setattr
            MetadataCatalog.get(dataset_name).evaluator_type = "interactive_grounding"

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=''):
        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "min_size_test": cfg['INPUT']['MIN_SIZE_TEST'],
            "max_size_test": cfg['INPUT']['MAX_SIZE_TEST'],
            "grounding": cfg['STROKE_SAMPLER']['EVAL']['GROUNDING'],
        }
        return ret

    def get_pascal_labels(self, ):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
                (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict['file_name']
        image = Image.open(file_name).convert('RGB')

        dataset_dict['width'] = image.size[0]
        dataset_dict['height'] = image.size[1]

        # if not self.is_train:
        image = transforms.ToTensor()(image)
        image = self.transform(image)
        # image = image.permute(2, 0, 1)

        class_ids = [ann_dict['category_id'] for ann_dict in dataset_dict['annotations']]
        instances = Instances(image.shape[-2:])
        mask_instances = Instances(image.shape[-2:])
        if 'inst_name' in dataset_dict.keys():
            inst_name = dataset_dict['inst_name']
            instances_mask = cv2.imread(inst_name)
            instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)

            objects_ids = dataset_dict['objects_ids']
            instances_mask_byid = [(instances_mask == idx).astype(np.int16) for idx in objects_ids]

            semseg_name = dataset_dict['semseg_name']
            semseg = self.encode_segmap(cv2.imread(semseg_name)[:, :, ::-1])
            # #class_names = [PASCAL_CLASSES[np.unique(semseg[instances_mask_byid[i].astype(np.bool)])[0].astype(
            # np.int32)-1] for i in range(len(instances_mask_byid))]

            _, h, w = image.shape
            masks = BitMasks(torch.stack([torch.from_numpy(
                cv2.resize(m.astype(np.float), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.bool)
            ) for m in instances_mask_byid]))
            mask_instances.gt_masks = masks
            # instances.gt_boxes = masks.get_bounding_boxes()

            for i in range(len(instances_mask_byid)):
                instances_mask_byid[i][instances_mask == self.ignore_id] = -1
            gt_masks_orisize = torch.stack([torch.from_numpy(m) for m in instances_mask_byid])
            dataset_dict['gt_masks_orisize'] = gt_masks_orisize  # (nm,h,w)
            

        instances.gt_boxes = torch.tensor([data_dict['bbox'] for data_dict in dataset_dict['annotations']])
        instances.gt_classes = torch.tensor(class_ids)
            

        # # To have information about the classes existing in this image for use in getting apropriate prompt/support
        # image
        classes_info = {}
        for item in dataset_dict['annotations']:
            if item['category_id'] not in classes_info:
                classes_info[item['category_id']] = 0
            classes_info[item['category_id']] += 1

        dataset_dict['instances'] = instances  # gt_masks, gt_boxes
        dataset_dict['mask_instances'] = mask_instances
        dataset_dict['image'] = image  # (3,h,w)
        ##dataset_dict['classes'] = class_names  # [prompt_engineering(x, topk=1, suffix='.') for x in class_names]
        dataset_dict['classes_info'] = classes_info
        return dataset_dict
