import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ExifTags
import numpy as np
import random
from config import common_config as config
import scipy.io as sio
import json

class GNHKDataset(Dataset):
    CHARS = config['chars']
    CHAR2LABEL = {char: i for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    def __init__(self, root_dir, mode='train', img_height=32, img_width=100):
        self.root_dir = root_dir
        self.split = mode
        self.img_height = img_height
        self.img_width = img_width
        self.image_paths = []
        self.annotation_paths = []
        self._load_data()

    def _load_data(self):
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
            self.image_paths = self.image_paths[int(0.1*len(self.image_paths)):int(0.95*len(self.image_paths))]
            self.annotation_paths = self.annotation_paths[int(0.1*len(self.annotation_paths)):int(0.95*len(self.annotation_paths))]
        elif self.split == 'val':
            self.image_paths = self.image_paths[:int(0.1*len(self.image_paths))]
            self.annotation_paths = self.annotation_paths[:int(0.1*len(self.annotation_paths))]

        # Extract word images and labels
        self.word_images = []
        self.word_labels = []
        for idx in range(len(self.image_paths)):
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
            image = image.convert('L')      
            
            # Load annotations from JSON
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)

            # Draw bounding boxes
            for annotation in annotations:
                polygon = annotation['polygon']
                label = annotation['text'].lower()
                if label != "%math%" and '£' not in label:
                    x_coordinates = [polygon['x0'],polygon['x1'],polygon['x2'],polygon['x3']]
                    y_coordinates = [polygon['y0'],polygon['y1'],polygon['y2'],polygon['y3']]
                    # Calculate minimum and maximum coordinates
                    xmin = min(x_coordinates)
                    ymin = min(y_coordinates)
                    xmax = max(x_coordinates)
                    ymax = max(y_coordinates)
                    # Append rectangle coordinates to the boxes list
                    word_image = image.crop((xmin, ymin, xmax, ymax))
                    self.word_images.append(word_image)
                    self.word_labels.append(label)  # Append word label


    def __len__(self):
        return len(self.word_images)

    def __getitem__(self, index):
        image = self.word_images[index]
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        text = self.word_labels[index].lower()
        target = [self.CHAR2LABEL[c] for c in text]
        target_length = [len(target)]

        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)
        return image, target, target_length



def gnhk_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths