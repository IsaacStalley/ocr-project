import os
import sys
import scipy.io
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath(r'C:\Users\IsaacStalley\Documents\GitHub\Mask_RCNN')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class SynthTextConfig(Config):
    """Configuration for training on the SynthText dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "synthtext"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + text

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class SynthTextDataset(utils.Dataset):

    def load_synthtext(self, dataset_dir, subset):
        """Load a subset of the SynthText dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. Assuming only one class, 'text'.
        self.add_class("text", 1, "text")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        total_images = len(imnames)
        if subset == "train":
            start_index = 0
            end_index = int(0.8 * total_images)
        elif subset == "val":
            start_index = int(0.8 * total_images)
            end_index = total_images

        # Load annotations
        annotations = scipy.io.loadmat(os.path.join(dataset_dir, "gt.mat"))

        imnames = annotations['imnames'][0]
        wordBB = annotations['wordBB'][0]
        charBB = annotations['charBB'][0]
        txt = annotations['txt'][0]

        # Add images and annotations
        for i in range(start_index, end_index):
            image_path = os.path.join(dataset_dir, imnames[i][0])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # Extract word-level bounding boxes
            word_bboxes = wordBB[i].transpose((2, 1, 0))

            for bbox in word_bboxes:
                polygons = [coord for point in bbox for coord in point]

            self.add_image(
                "text",
                image_id=imnames[i][0],  # using file name as unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons)
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "text":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)