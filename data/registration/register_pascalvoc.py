# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
from typing import List, Tuple, Union
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.io import loadmat

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_pascalvoc_instances", "register_pascalvoc_context"]

PASCAL_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)


PASCAL_VOC_BASE_CLASSES = (
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
        )

PASCAL_VOC_NOVEL_CLASSES = ("bird", "bus", "cow", "motorbike", "sofa")


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()


def load_pascalvoc_instances(name: str, dirname: str, mode: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    if mode == "Base":
        classes = PASCAL_VOC_BASE_CLASSES
    elif mode == "Novel":
        classes = PASCAL_VOC_NOVEL_CLASSES
    else:
        classes = PASCAL_CLASSES

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        inst_path = os.path.join(dirname, "SegmentationObject", "{}.png".format(fileid))
        semseg_path = os.path.join(dirname, "SegmentationClass", "{}.png".format(fileid))

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        # check and see if instance segmentation also exists and append it to the annotations
        if os.path.exists(inst_path):
            instances_mask = cv2.imread(inst_path)
            instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)

            objects_ids = np.unique(instances_mask)
            objects_ids = [x for x in objects_ids if x != 0 and x != 220]

            slice_size = 5
            for i in range(0, len(objects_ids), slice_size):
                r2 = {
                    "inst_name": inst_path,
                    "semseg_name": semseg_path,
                    "objects_ids": objects_ids[i:i+slice_size],
                }
                r.update(r2)
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": PASCAL_CLASSES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        #
        if len(instances) > 0:
            r["annotations"] = instances
            dicts.append(r)
        # elif split == "val":
        #     r["annotations"] = instances
        #     dicts.append(r)
    return dicts


def register_pascalvoc_context(name, dirname, mode, split):
    DatasetCatalog.register("{}_{}".format(name, mode), lambda: load_pascalvoc_instances(name, dirname, mode, split))
    MetadataCatalog.get("{}_{}".format(name, mode)).set(
        dirname=dirname,
        thing_dataset_id_to_contiguous_id={},
    )


def register_all_sbd(root):
    SPLITS = [
        #("pascalvoc_val", "PascalVOC", "Point", "val"),
        #("pascalvoc_val", "PascalVOC", "Scribble", "val"),
        #("pascalvoc_val", "PascalVOC", "Polygon", "val"),
        ("pascalvoc_val", "PascalVOC", "Base", "val"),
        #("pascalvoc_val", "PascalVOC", "Novel", "val"),
        ("pascalvoc_train", "PascalVOC", "Base", "train"),
        ("pascalvoc_uptrain", "PascalVOC", "All", "train"),
        ("pascalvoc_upval", "PascalVOC", "All", "val"),
    ]

    for name, dirname, mode, split in SPLITS:
        register_pascalvoc_context(name, os.path.join(root, dirname), mode, split)
        MetadataCatalog.get("{}_{}".format(name, mode)).evaluator_type = "pascal"


_root = os.getenv("DATASET", "datasets")
register_all_sbd(_root)
