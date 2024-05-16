import os
import numpy as np
from PIL import Image, ExifTags, ImageDraw
import scipy.io as sio
import cv2
import json
import copy

class GnhkTextDataset:
    def __init__(self, root_dir, transform=None, target_transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.image_paths = []
        self.annotation_paths = []
        self.class_names = ('BACKGROUND', 'text')
        self._load_annotations()

    def _load_annotations(self):
        # Iterate through files in the dataset directory
        for file_name in os.listdir(self.root_dir):
            if file_name.endswith('.jpg'):
                # Found an image file, add to image_paths
                self.image_paths.append(os.path.join(self.root_dir, file_name))
                
                # Corresponding annotation file
                annotation_name = file_name.replace('.jpg', '.json')
                annotation_path = os.path.join(self.root_dir, annotation_name)
                self.annotation_paths.append(annotation_path)
        
        if self.split == 'train':
            self.image_paths = self.image_paths[:int(0.9*len(self.image_paths))]
            self.annotation_paths = self.annotation_paths[:int(0.9*len(self.annotation_paths))]
        elif self.split == 'val':
            self.image_paths = self.image_paths[int(0.9*len(self.image_paths)):]
            self.annotation_paths = self.annotation_paths[int(0.9*len(self.annotation_paths)):]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        if image_path is None:
            raise IOError(f"failed to find {image_path}")
            
        image = Image.open(image_path)
        
        if image is None or image.size == 0:
            raise IOError(f"invalid/corrupt image {image_path}")

        exif = image._getexif()
        if exif is not None:
            orientation = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
            if orientation in exif:
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
        image = image.convert('RGB')      
        
        # Load annotations from JSON
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        boxes = []
        labels = []
        # Draw bounding boxes
        for annotation in annotations:
            polygon = annotation['polygon']
            x_coordinates = [polygon['x0'],polygon['x1'],polygon['x2'],polygon['x3']]
            y_coordinates = [polygon['y0'],polygon['y1'],polygon['y2'],polygon['y3']]
            # Calculate minimum and maximum coordinates
            xmin = min(x_coordinates)
            ymin = min(y_coordinates)
            xmax = max(x_coordinates)
            ymax = max(y_coordinates)
            # Append rectangle coordinates to the boxes list
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        image = np.asarray(image)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        
        return image, boxes, labels