import os
import numpy as np
from PIL import Image, ImageDraw
import scipy.io as sio
import cv2
import copy

class IAMTextDataset:
    def __init__(self, root_dir, transform=None, target_transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.image_paths = []
        self.annotations = {}
        self.class_names = ('BACKGROUND', 'line')
        self._load_annotations()

    def _load_annotations(self):
        # Iterate through files in the dataset directory
        image_path = os.path.join(self.root_dir, 'formsA-D')
        for file_name in os.listdir(image_path):
            if file_name.endswith('.png'):
                self.image_paths.append(file_name)
                self.annotations[file_name] = []

        # Load annotations from words.txt
        annotation_path = os.path.join(self.root_dir, 'ascii/words.txt')
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        # Process annotations
        for line in lines:
            if not line.startswith('#'):
                parts = line.strip().split()
                image_id = parts[0].split("-")[0] + "-" + parts[0].split("-")[1]
                image_path = f'{image_id}.png'
                if image_path in self.annotations:
                    self.annotations[image_path].append(parts[3:7])  # Append bounding box coordinates

        if self.split == 'train':
            self.image_paths = self.image_paths[:int(0.9*len(self.image_paths))]
        elif self.split == 'val':
            self.image_paths = self.image_paths[int(0.9*len(self.image_paths)):]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, image_id):
        image_id = self.image_paths[image_id]
        image_path = os.path.join(self.root_dir, 'formsA-D')
        image_path = os.path.join(image_path, image_id)

        # Load image
        image = Image.open(image_path)
        if image is None or image.size == 0:
            raise IOError(f"invalid/corrupt image {image_id}")

        # Find highest and lowest bounding boxes
        highest_y_min = float('inf')
        lowest_y_max = -1
        for bbox in self.annotations[image_id]:
            y_min = int(bbox[1])
            y_max = y_min + int(bbox[3])
            highest_y_min = min(highest_y_min, y_min)
            lowest_y_max = max(lowest_y_max, y_max)
        threshold=10
        # Add threshold
        highest_y_min = max(0, highest_y_min - threshold)
        lowest_y_max = min(image.height, lowest_y_max + threshold)

        # Crop image
        image = image.crop((0, highest_y_min, image.width, lowest_y_max))
        boxes = []
        labels = []
        # Draw bounding boxes
        for bbox in self.annotations[image_id]:
            x_min, y_min, width, height = map(int, bbox)
            if x_min < 0 or y_min < 0 or width < 0 or height < 0:
                continue 

            x_max = x_min + width
            y_max = y_min + height
            box = [x_min, y_min - highest_y_min, x_max, y_max - highest_y_min]
            boxes.append(box)
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