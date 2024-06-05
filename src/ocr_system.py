import sys
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2
import utils
from config import ssd_config
from config import crnn_config
sys.path.append('../models/pytorch_ssd')
sys.path.append('../models/crnn_pytorch')

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from src.model import CRNN
from src.ctc_decoder import ctc_decode

class SingleImageDataset(Dataset):
    def __init__(self, images, img_height, img_width):
        self.images = images
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)
        return image

class TextRecognitionSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        label_path = ssd_config['label_path']
        model_path = ssd_config['model_path']
        # Initialize the first model (Predictor)
        class_names = [name.strip() for name in open(label_path).readlines()]
        net = create_vgg_ssd(len(class_names), is_test=True)
        net.load(model_path)
        self.predictor = create_vgg_ssd_predictor(net, candidate_size=200)
        CHARS = crnn_config['chars']
        CHAR2LABEL = {char: i for i, char in enumerate(CHARS)}
        self.LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
        num_class = len(self.LABEL2CHAR) #+ 1
        self.crnn = CRNN(1, crnn_config['img_height'], crnn_config['img_width'], num_class,
                    map_to_seq_hidden=crnn_config['map_to_seq_hidden'],
                    rnn_hidden=crnn_config['rnn_hidden'],
                    leaky_relu=crnn_config['leaky_relu'])
        self.crnn.load_state_dict(torch.load(crnn_config['model_path'], map_location=self.device))
        self.crnn.to(self.device)

    def process_image(self, image_path):
        # Load the input image
        orig_image = cv2.imread(image_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        # Use Predictor to get bounding boxes
        boxes, labels, probs = self.predictor.predict(image, -1, 0.2)

        boxes = utils.filter_overlapping_boxes(boxes)
        boxes = utils.filter_small_boxes(boxes, image)
        boxes, line_indeces = utils.sort_boxes(boxes, image)
        boxes = utils.add_box_buffer(boxes, image)

        word_images = []
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            word_image = orig_image[y_min:y_max, x_min:x_max]
            word_images.append(word_image)

        # Prepare the word images for CRNN
        word_dataset = SingleImageDataset(word_images, crnn_config['img_height'], crnn_config['img_width'])
        word_loader = DataLoader(dataset=word_dataset, batch_size=1, shuffle=False)

        # Predict the text using CRNN
        preds = self.predict(word_loader)
        words = [''.join(word) for word in preds]

        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(orig_image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 4)
            cv2.putText(orig_image, words[i],
                (x_min + 5, y_min + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
        path = "run_ssd_example_output.jpg"
        cv2.imwrite(path, orig_image)

        return words, line_indeces

    def predict(self, dataloader):
        all_preds = []
        with torch.no_grad():
            for data in dataloader:
                device = 'cuda' if next(self.crnn.parameters()).is_cuda else 'cpu'
                images = data.to(device)
                logits = self.crnn(images)
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                preds = ctc_decode(log_probs, method=crnn_config['decode_method'], beam_size=crnn_config['beam_size'],
                               label2char=self.LABEL2CHAR)
                all_preds += preds
        return all_preds