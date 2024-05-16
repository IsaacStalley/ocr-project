import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from config import common_config as config
import random

class SynthTigerDataset(Dataset):
    CHARS = config['chars']
    CHAR2LABEL = {char: i for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir, img_height=32, img_width=100, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width
        self.image_paths = []
        self.labels = []

        labels_file = os.path.join(self.root_dir, 'gt.txt')
        with open(labels_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                self.image_paths.append(os.path.join(self.root_dir, parts[0]))
                self.labels.append(parts[1])
        if self.mode == 'train':
            self.image_paths = self.image_paths[:int(0.99*len(self.image_paths))]
            self.labels = self.labels[:int(0.99*len(self.labels))]
        elif self.mode == 'val':
            self.image_paths = self.image_paths[int(0.99*len(self.image_paths)):]
            self.labels = self.labels[int(0.99*len(self.labels)):]

    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        index = random.randint(0, len(self.image_paths)-1)
        path = self.image_paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        label_text = self.labels[index].lower()
        target = [self.CHAR2LABEL[c] for c in label_text]
        target_length = [len(target)]

        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)

        return image, target, target_length


def synthtiger_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths