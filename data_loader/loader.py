from .utils import return_bounding_boxes
import torch
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2

load_dotenv()

class ECGDataset(Dataset):
    def __init__(self, image_path, log_path, max_leads=13, padded_image_size=(224, 224)):
        with open(log_path, 'r') as f:
            # Load the JSON data
            self.logs = json.load(f)['samples']

        self.image_path = image_path
        self.max_leads = max_leads
        self.padded_image_size = padded_image_size

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        log = self.logs[idx]
        image_path = log['image_name']
        image = plt.imread(f"{self.image_path}/{image_path}.jpg")

        # Resize the image to the specified size
        image = cv2.resize(image, self.padded_image_size)

        # Pad the image if necessary
        if image.shape[0] < self.padded_image_size[0] or image.shape[1] < self.padded_image_size[1]:
            pad_height = max(self.padded_image_size[0] - image.shape[0], 0)
            pad_width = max(self.padded_image_size[1] - image.shape[1], 0)
            image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

        # Crop the image if necessary
        if image.shape[0] > self.padded_image_size[0] or image.shape[1] > self.padded_image_size[1]:
            image = image[:self.padded_image_size[0], :self.padded_image_size[1], :]

        bounding_boxes = return_bounding_boxes(image, log)
        bounding_boxes = np.array(bounding_boxes)

        # Pad the bounding boxes to (13, 4)
        padded_bounding_boxes = np.pad(bounding_boxes, ((0, 13 - len(bounding_boxes)), (0, 0)), mode='constant')
        bounding_boxes = padded_bounding_boxes[:13]

        return image, bounding_boxes

if __name__ == '__main__':
    # path = f"{os.getenv('signal_dataset_path')}image_dataset_v4.0/"
    path = "datasets/image_dataset_v4.0/"
    dataset = ECGDataset(image_path=path, log_path=f"{path}logs.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for images, bounding_boxes in dataloader:
        # Process the batch of images and bounding boxes
        print(images.shape, bounding_boxes.shape)
