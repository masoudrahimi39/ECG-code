import os
import cv2
from PIL import Image

import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

# DATA_DIR = "./data/CamVid/"

x_train_dir = r'D:/Education/4_ECG_digitization/ECG-code/datasets/image_dataset_v7.0/segmentation/train/images'
y_train_dir = r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\segmentation\train\labels'

x_valid_dir = r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\segmentation\val\images'
y_valid_dir = r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\segmentation\val\labels'

x_test_dir = r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\segmentation\test\images'
y_test_dir = r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\segmentation\test\labels'

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)

    """

    CLASSES = [
        "signal",
        "unlabelled",
    ]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
    ):
        self.image_ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        # self.masks_fps = [os.path.join(masks_dir, os.path.splitext(image_id)[0] + '.bmp') for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
    
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR-->RGB

        mask = Image.open(self.masks_fps[i]).convert('L')  # load single channel
        mask = np.array(mask) 
        assert np.all(np.isin(mask, [0, 1])), "Mask contains non-binary values"

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        
        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)

    def __len__(self):
        return len(self.ids)


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name == "image":
            plt.imshow(image.transpose(1, 2, 0))
        else:
            plt.imshow(image)
    plt.show()


# training set images augmentation
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480),
    ]
    return A.Compose(test_transform)


if __name__ == '__main__':
    dataset = Dataset(x_train_dir, y_train_dir, 
                      augmentation=get_training_augmentation(), 
                      classes=["signal"])
    
    for i in range(3):
        image, mask = dataset[3]
        visualize(image=image, mask=mask.squeeze())
    