# dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt

# there was an error in concatenation of unet with the upsampled vector (https://stackoverflow.com/questions/66573636/keras-unet-a-concatenate-layer-requires-inputs-with-matching-shapes-except-fo)
# to handle this error, we added factor of 2 for 5 times to the input and mask image since after 5 layers of down sampling we stll gonna have
# two images with identical shape for concatenation 
dividable = 32 

class SimDataset(Dataset):
    def __init__(self, path, input_transform=None, label_transform=None):
        self.path = path
        self.input_path = os.path.join(path, "input")
        self.label_path = os.path.join(path, "label")
        _, _, self.image_files = next(os.walk(self.input_path))
        self.input_transform = input_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = plt.imread(os.path.join(self.input_path, self.image_files[idx]))
        mask = plt.imread(os.path.join(self.label_path, self.image_files[idx]))
        if self.input_transform:
            image = self.input_transform(image)
        if self.label_transform:
            mask = self.label_transform(mask)

        return [image, mask]

def pad_collate(batch):
    input_batch = [b[0] for b in batch]
    label_batch = [b[1] for b in batch]

    max_height = max([img.size(1) for img in input_batch])
    max_width = max([img.size(2) for img in input_batch])

    max_height = ((max_height-1)//dividable+1)*dividable
    max_width = ((max_width-1)//dividable+1)*dividable

    padded_input = []
    padded_label = []

    for input_image, label_image in zip(input_batch, label_batch):
        padding = (0, 0, max_width - input_image.size(2), max_height - input_image.size(1))
        tmp1 = transforms.functional.pad(input_image, padding)
        tmp2 = transforms.functional.pad(label_image, padding)

        padded_input.append(tmp1)
        padded_label.append(tmp2)

    return torch.stack(padded_input), torch.stack(padded_label)

# Transformation for train/val
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # imagenet normalization
])

# Define the transformation
label_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Grayscale(),  # Convert image to grayscale with 3 channels
])

# Create datasets
train_set = SimDataset("datasets/image_dataset_v4.0/refinement/train", input_transform=input_transform, label_transform=label_transform)
val_set = SimDataset("datasets/image_dataset_v4.0/refinement/val", input_transform=input_transform, label_transform=label_transform)

# Dataloaders
dataloaders = {
    'train': DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0, collate_fn=pad_collate),
    'val': DataLoader(val_set, batch_size=2, shuffle=True, num_workers=0, collate_fn=pad_collate)
}
