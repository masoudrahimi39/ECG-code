import os
import cv2
from PIL import Image
import random

import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

'https://segmentation-modelspytorch.readthedocs.io/en/latest/index.html#'
'https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/binary_segmentation_intro.ipynb'

def scale_up_img_size(image, target_height, divisible_by=32):
    '''Calculate new dimensions for image that are divisible by 32 as it is necessary for UNet input to be divisible by 32'''

    # print(f"Input image shape: {image.shape}")
    # print(f"Target height: {target_height}")

    # target_height = ((image.shape[0] + divisible_by-1) // divisible_by) * divisible_by  # Round up to nearest multiple of divisible_by
    target_width = ((image.shape[1] + divisible_by-1) // divisible_by) * divisible_by

    # Calculate padding amounts
    pad_height = target_height - image.shape[0]
    pad_width = target_width - image.shape[1]
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # print(f"Calculated pad_height: {pad_height}")
    # print(f"Calculated pad_width: {pad_width}")
    # print(f"Padding values - top: {pad_top}, bottom: {pad_bottom}, left: {pad_left}, right: {pad_right}")

    if len(image.shape) == 2:
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    elif len(image.shape) == 3:
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    return image

def resize_and_crop(image, mask, target_height=160, target_width=320):
    mask[mask>1] = 0

    height, width = image.shape[:2]

    # Determine the scaling factor to resize the smaller dimension to 320
    if width/target_width < height/target_height:
        new_width = target_width
        new_height = target_height
        # new_height = int((height / width) * target_width)
    else:
        new_height = target_height
        new_width = int((width / height) * target_height)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # Determine the random crop coordinates
    x_max = new_width - target_width
    y_max = new_height - target_height

    if x_max > 0:
        x_start = random.randint(0, x_max)
    else:
        x_start = 0

    if y_max > 0:
        y_start = random.randint(0, y_max)
    else:
        y_start = 0

    cropped_image = resized_image[y_start:y_start + target_height, x_start:x_start + target_width]
    cropped_mask = resized_mask[y_start:y_start + target_height, x_start:x_start + target_width]

    return cropped_image, cropped_mask

class AddVerticalLines(A.DualTransform):
    def __init__(self, num_lines=5, line_width=1, always_apply=False, p=0.5):
        super(AddVerticalLines, self).__init__(always_apply, p=p)
        self.num_lines = num_lines  # تعداد خطوط
        self.line_width = line_width  # عرض خطوط

    def apply(self, image, **params):
        h, w, c = image.shape
        for _ in range(self.num_lines):
            x = random.randint(0, w - self.line_width)
            image[:, x:x + self.line_width, :] = 0  # خط سیاه عمودی
        return image

    def apply_to_mask(self, mask, **params):
        return mask  # mask بدون تغییر می‌ماند

class OverlaySignal(A.DualTransform):
    def __init__(self, signal_dir, crop_height=50, always_apply=False, p=0.5):
        super(OverlaySignal, self).__init__(always_apply, p=p)
        self.signal_dir = signal_dir  # پوشه حاوی تصاویر BMP سیگنال
        self.crop_height = crop_height  # ارتفاع بخشی که برش داده می‌شود
        self.signal_files = [f for f in os.listdir(signal_dir) if f.endswith('.bmp')]

    def apply(self, image, **params):
        # انتخاب تصادفی یک تصویر سیگنال
        signal_file = random.choice(self.signal_files)
        signal_path = os.path.join(self.signal_dir, signal_file)
        signal = cv2.imread(signal_path, cv2.IMREAD_GRAYSCALE)  # خواندن به صورت تک‌کاناله
        signal = np.array(signal)

        # برش تصادفی بخشی از سیگنال
        h, w = signal.shape
        # crop_height = random.randint(5, self.crop_height)
        # crop_y = random.randint(0, h - self.crop_height)
        crop_height = random.randint(int(0.12*h), int(0.7*h))
        if random.random() < 0.5:
          cropped_signal = signal[: crop_height, :]
          cropped_signal = cropped_signal[::-1, :]
        else:
          cropped_signal = signal[-crop_height:, :]

        crop_width = random.randint(int(0.05*w), int(0.1*w))
        cropped_signal = cropped_signal[:, crop_width:-crop_width]

        if cropped_signal.shape[1]>image.shape[1]:
          cropped_signal = cropped_signal[: ,:image.shape[1]]

        # تغییر اندازه سیگنال برش‌خورده به عرض تصویر ورودی
        cropped_signal = cv2.resize(cropped_signal, (image.shape[1], self.crop_height))

        x, y = np.where(cropped_signal > 0)
        image[x, y, :] = 0

        # # انتخاب تصادفی بالا یا پایین تصویر
        # if random.random() < 0.5:
        #     # image[:self.crop_height, :, 1] = np.where(cropped_signal > 0, 0, image[:self.crop_height, :, 0])
        #     # image[:self.crop_height, :, 2] = np.where(cropped_signal > 0, 0, image[:self.crop_height, :, 0])
        # else:
        #     image[-self.crop_height:, :, 0] = np.where(cropped_signal > 0, 0, image[-self.crop_height:, :, 0])
        #     image[-self.crop_height:, :, 1] = np.where(cropped_signal > 0, 0, image[-self.crop_height:, :, 0])
        #     image[-self.crop_height:, :, 2] = np.where(cropped_signal > 0, 0, image[-self.crop_height:, :, 0])

        return image

    def apply_to_mask(self, mask, **params):
        return mask  # mask بدون تغییر می‌ماند


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)

    """

    CLASSES = ["signal"]

    def __init__(
        self,
        images_dir,
        masks_dir,
        max_height,      # maximum height of images. as you know, we crop the leads and it will have different height
        classes=None,
        augmentation=None,
    ):
        self.max_height = max_height
        self.image_ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)

        # self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        # self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        self.images_fps = [] # [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [] # [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        for image_id in self.image_ids:
          mask_path = os.path.join(masks_dir, image_id.replace(".png", ".bmp"))
          if not os.path.exists(mask_path):
            continue
          self.images_fps.append(os.path.join(images_dir, image_id))
          self.masks_fps.append(mask_path)

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        # self.augmentation = augmentation
        # self.augmentation = A.Compose([
        #   OverlaySignal(signal_dir='/content/drive/MyDrive/dataset/image_dataset_v9.0/segmentation/train/labels', crop_height=40, p=0.3),
        #   # CustomResizeAndCrop(height=128, width=320, always_apply=True, p=1.0),
        #   AddVerticalLines(num_lines=2, line_width=2, p=0.3),
        #   A.VerticalFlip(p=0.4),
        #   A.HorizontalFlip(p=0.4),
        #   A.Rotate(limit=5, p=0.2),
        #   # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        #   # A.HueSaturationValue(p=0.5),
        #   # A.ToGray(p=0.2),
        #   A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        #   A.GaussianBlur(blur_limit=(1, 5), p=0.2),
        #   # A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
        # ])
        self.augmentation = A.Compose([
          A.VerticalFlip(p=0.5),
          A.HorizontalFlip(p=0.5),
          OverlaySignal(signal_dir='/content/drive/MyDrive/dataset/image_dataset_v9.0/segmentation/train/labels', crop_height=30, p=0.4),
          A.Rotate(limit=5, p=0.3),
          AddVerticalLines(num_lines=2, line_width=2, p=0.3),
          A.GaussNoise(var_limit=(1.0, 4.0), p=0.2),
          A.GaussianBlur(blur_limit=(1, 5), p=0.2),
        ])

    def __getitem__(self, i):
        selected_size = (128, 320)

        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR-->RGB
        image = np.array(image)

        # image = cv2.resize(image, selected_size)

        # if image.shape[0] > self.max_height:
        #     # print(f"Skipping image {self.images_fps[i]} as its height {image.shape[0]} is greater than maximum height{self.max_height}")
        #     return self.__getitem__((i + 1) % len(self))

        # image = scale_up_img_size(image, divisible_by=32, target_height=self.max_height) # scale up image width and height

        mask = Image.open(self.masks_fps[i]).convert('L')  # load single channel
        mask = np.array(mask)
        # mask = cv2.resize(mask, selected_size)


        # mask = scale_up_img_size(mask, divisible_by=32, target_height=self.max_height)
        image, mask = resize_and_crop(image, mask, target_height=selected_size[0], target_width=selected_size[1])

        # assert np.all(np.isin(mask, [0, 1])), "Mask contains non-binary values"

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")
        mask = 1 - mask                                    # convert 1 to 0 and 0 to 1 since the background should be 0 and signal should be 1

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        
        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)

    def __len__(self):
        return len(self.images_fps)


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
    x_train_dir = 'datasets/image_dataset_v7.0/segmentation/train/images'
    y_train_dir = 'datasets/image_dataset_v7.0/segmentation/train/labels'

    dataset = Dataset(x_train_dir, y_train_dir, 
                      max_height=320,
                    #   augmentation=get_training_augmentation(), 
                      augmentation=None,
                      classes=["signal"])
    
    for i in range(1):
        image, mask = dataset[i]

        # visualize(image=image, mask=mask.squeeze())
    