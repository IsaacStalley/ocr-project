import os
import json
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2
import copy
import random

class ImgurDataset:
    def __init__(self, root_dir, transform=None, target_transform=None, split='train'):
        self.dataset_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.class_names = ('BACKGROUND', 'line')
        self._load_annotations()
        self._load_index_ids()

    def _load_annotations(self):
        with open(os.path.join(self.dataset_dir, "dataset_info/imgur5k_annotations.json"), "r") as file:
            data = json.load(file)
        
        self.index_to_ann_map = data.get("index_to_ann_map", {})
        self.ann_id = data.get("ann_id", {})
        self.index_to_image_info = data.get("index_id", {})

    def _load_index_ids(self):
        if self.split == 'train':
            with open(os.path.join(self.dataset_dir, "dataset_info/train_index_ids.lst"), "r") as file:
                self.index_ids = [line.strip() for line in file]
        elif self.split == 'val':
            with open(os.path.join(self.dataset_dir, "dataset_info/val_index_ids.lst"), "r") as file:
                self.index_ids = [line.strip() for line in file]

    def __len__(self):
        return len(self.index_ids)

    def __getitem__(self, index):
        while True:
            index_id = self.index_ids[index]
            image_info = self.index_to_image_info.get(index_id)
            if image_info is None:
                print("Image not found for index:", index)
                index = index + 1
                index_id = self.index_ids[index]
                image_info = self.index_to_image_info.get(index_id)
            
            image_path = os.path.join(self.dataset_dir, image_info.get("image_path"))
            if image_path is None:
                print("Image path not found for index:", index)
                return
            if os.path.exists(image_path):
                break
            index = index + 1
        
        image = Image.open(image_path)
        if image is None or image.size == 0:
            raise IOError(f"invalid/corrupt image {index_id}")
        
        ann_ids = self.index_to_ann_map.get(index_id, [])
        boxes = []
        labels = []
        for ann_id in ann_ids:
            annotation = self.ann_id.get(ann_id)
            if annotation is None:
                continue
            
            bounding_box = annotation.get("bounding_box")
            word = annotation.get("word")
            if bounding_box is None:
                continue
            
            bounding_box = json.loads(bounding_box)
            # Extract bounding box coordinates
            xc, yc, w, h, a = bounding_box
            boxes.append([xc - w/2, yc - h/2, xc + w/2, yc + h/2])
            labels.append(1)
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        image = image.convert('RGB')
        image = np.asarray(image)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        
        return image, boxes, labels