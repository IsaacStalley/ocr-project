import os
import numpy as np
import scipy.io as sio
import cv2
import copy
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw
import random

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
    
    # Function to randomly adjust exposure
    def random_exposure(self, image):
        factor = random.uniform(0.2, 1.2)  # adjust exposure by a factor between 0.5 and 1.5
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    # Function to randomly invert colors
    def random_invert(self, image):
        if random.random() < 0.1:  # 50% chance of inverting colors
            return ImageOps.invert(image)
        else:
            return image

    # Function to randomly change colors
    def random_color_jitter(self, image):
        r, g, b = image.split()
        r = np.array(r)
        g = np.array(g)
        b = np.array(b)

        r_shift = random.randint(-50, 50)
        g_shift = random.randint(-50, 50)
        b_shift = random.randint(-50, 50)

        r = np.clip(r + r_shift, 0, 255)
        g = np.clip(g + g_shift, 0, 255)
        b = np.clip(b + b_shift, 0, 255)

        r = Image.fromarray(r).convert('L')
        g = Image.fromarray(g).convert('L')
        b = Image.fromarray(b).convert('L')

        return Image.merge('RGB', (r, g, b))


    # Function to randomly apply Gaussian blur
    def random_blur(self, image):
        kernel_size = random.randint(0, 2) * 2 + 1  # Choose an odd kernel size between 3 and 11
        return image.filter(ImageFilter.GaussianBlur(kernel_size))

    # Function to randomly add noise to the image
    def random_noise(self, image):
        width, height = image.size
        noise = np.random.randint(0, 10, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(np.clip(np.array(image) + noise, 0, 255).astype(np.uint8))

    # Apply random transformations to the image
    def apply_random_transformations(self, image):
        image = self.random_exposure(image)
        image = self.random_invert(image)
        #image = self.random_color_jitter(image)
        #image = self.random_blur(image)
        #image = self.random_noise(image)
        return image

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
            if y_max < 0 or y_min < 0:
                continue
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
        image = self.apply_random_transformations(image)
        image = np.asarray(image)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels