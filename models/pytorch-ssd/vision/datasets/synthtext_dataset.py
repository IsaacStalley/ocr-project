import os
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2
import copy

class SynthTextDataset:
    def __init__(self, root_dir, transform=None, target_transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self._load_annotations()

    def _load_annotations(self):
        # Load annotations from gt.mat file
        annotations = sio.loadmat(os.path.join(self.root_dir, "gt.mat"))

        self.image_paths = annotations['imnames'][0]
        self.word_bounding_boxes = annotations['wordBB'][0]
        self.character_bounding_boxes = annotations['charBB'][0]
        self.text = annotations['txt'][0]
        self.class_names = ('BACKGROUND', 'text')

        if self.split == 'train':
            self.image_paths = self.image_paths[:int(0.01*len(self.image_paths))]
        elif self.split == 'val':
            self.image_paths = self.image_paths[int(0.99*len(self.image_paths)):]
            self.word_bounding_boxes = self.word_bounding_boxes[int(0.99*len(self.word_bounding_boxes)):]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_paths[idx][0])
        if image_path is None:
            raise IOError(f"failed to find {image_path}")
            
        image = cv2.imread(str(image_path))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None or image.size == 0:
            raise IOError(f"invalid/corrupt image {image_path}")

        # Retrieve bounding boxes and text for the image
        word_bboxes = copy.copy(self.word_bounding_boxes[idx])
        if len(np.shape(word_bboxes)) == 2:
            word_bboxes = np.array([word_bboxes])
            word_bboxes = np.transpose(word_bboxes, (1, 2, 0))
        word_bboxes = np.transpose(word_bboxes, (2, 1, 0))

        boxes = []
        labels = []
        for bbox in word_bboxes:
            # Extract X and Y coordinates separately
            x_coordinates = [point[0] for point in bbox]
            y_coordinates = [point[1] for point in bbox]
            
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

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        
        return image, boxes, labels