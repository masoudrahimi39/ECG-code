import os
import numpy as np
import cv2
import sys
from pathlib import Path
from PIL import Image
sys.path.append("./model/digitization/ecg-digitize")

import ecgdigitize
from ecgdigitize.image import openImage
from scripts.signal_extraction import get_image_paths, get_final_signal

from model.segmentation.unet.inference import SegmentationInference

segmentaion_input_path = 'datasets/image_dataset_v8.0/segmentation/val/images'
digitization_input_path = 'datasets/image_dataset_v8.0/digitization/val/images'
digitization_output_path = 'datasets/image_dataset_v8.0/digitization/val/labels'
output_path = 'model/digitization/results'

model_path = "model/segmentation/unet/unet-segmentation-v31-1.ckpt"
segmenter = SegmentationInference(model_path)

for seg_in in get_image_paths(segmentaion_input_path):
    # seg_in = "/Users/rezakarbasi/PersonalFiles/Projects/4_OtherProj/ECG-proj/code/ECG/datasets/image_dataset_v8.0/segmentation/train/images/477_3by4_lead_3.png"
    digi_in = os.path.join(digitization_input_path, f"{Path(seg_in).stem}.png")
    digi_out = os.path.join(digitization_output_path, f"{Path(seg_in).stem}.json")
    mask_name = Path(seg_in).stem + "-mask.png"

    print(seg_in)
    print(digi_in)
    print(mask_name)

    if not (os.path.exists(digi_in) and os.path.exists(digi_out)):
        continue

    mask = segmenter(seg_in)
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    Image.fromarray(mask*255).convert('RGB').save(mask_name)
    x_vals, y_vals, corr_mask = get_final_signal(image_path=mask_name, label_path=digi_out, output_path=output_path, save_image=False)
    x_vals, y_vals, corr_digi = get_final_signal(image_path=digi_in, label_path=digi_out, output_path=output_path, save_image=False)
    print(f"->segmentaion->digitization: {corr_mask}")
    print(f"->digitization: {corr_digi}")

    # break