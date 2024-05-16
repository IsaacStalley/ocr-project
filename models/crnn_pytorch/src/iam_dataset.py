import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw
import numpy as np
import random
from config import common_config as config

class IAMDataset(Dataset):
    CHARS = config['chars']
    CHAR2LABEL = {char: i for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    def __init__(self, root_dir, mode='train', img_height=32, img_width=100):
        self.root_dir = root_dir
        self.split = mode
        self.img_height = img_height
        self.img_width = img_width
        self.image_paths = []
        self.annotations = {}
        self._load_data()

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

    def _load_data(self):
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
                    self.annotations[image_path].append((parts[3:7], parts[-1]))  # Append bounding box coordinates and word label
        
        if self.split == 'train':
            self.image_paths = self.image_paths[:int(0.9*len(self.image_paths))]
        elif self.split == 'val':
            self.image_paths = self.image_paths[int(0.9*len(self.image_paths)):]

        # Extract word images and labels
        self.word_images = []
        self.word_labels = []
        for image_path in self.image_paths:
            image = Image.open(os.path.join(self.root_dir, 'formsA-D', image_path)).convert('L')
            annotations = self.annotations[image_path]
            for annotation, label in annotations:
                # Extract bounding box coordinates
                x_min, y_min, width, height = map(int, annotation)
                if x_min < 0 or y_min < 0 or width < 0 or height < 0:
                    continue 

                x_max = x_min + width
                y_max = y_min + height
                assert x_min < x_max
                # Crop word image using bounding box coordinates
                word_image = image.crop((x_min, y_min, x_max, y_max))
                self.word_images.append(word_image)
                self.word_labels.append(label)  # Append word label

    def __len__(self):
        return len(self.word_images)

    def __getitem__(self, index):
        image = self.word_images[index]
        image = self.apply_random_transformations(image)
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



def iam_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths