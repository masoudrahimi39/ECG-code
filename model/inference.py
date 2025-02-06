import os
import sys
from pathlib import Path
from PIL import Image
sys.path.append("./model/digitization/ecg-digitize")

import ecgdigitize
from ecgdigitize.image import openImage
from scripts.signal_extraction import get_image_paths, get_final_signal

from model.segmentation.unet.inference import SegmentationInference

segmentaion_input_path = 'datasets/image_dataset_v7.0/segmentation/train/images'
digitization_input_path = 'datasets/image_dataset_v7.0/digitization/train/images'
digitization_output_path = 'datasets/image_dataset_v7.0/digitization/train/labels'
output_path = 'model/digitization/results'

model_path = "model/segmentation/unet/unet-segmentation.pth"
segmenter = SegmentationInference(model_path)

for seg_in in get_image_paths(segmentaion_input_path):
    digi_in = os.path.join(digitization_input_path, f"{Path(seg_in).stem}.png")
    digi_out = os.path.join(digitization_output_path, f"{Path(seg_in).stem}.json")

    if not (os.path.exists(digi_in) and os.path.exists(digi_out)):
        continue

    mask = segmenter(seg_in)
    Image.fromarray(mask*255).convert('RGB').save("./mask.jpg")
    x_vals, y_vals, corr_mask = get_final_signal(image_path="./mask.jpg", label_path=digi_out, output_path=output_path, save_image=False)
    x_vals, y_vals, corr_digi = get_final_signal(image_path=digi_in, label_path=digi_out, output_path=output_path, save_image=False)
    print(f"->segmentaion->digitization: {corr_mask}")
    print(f"->digitization: {corr_digi}")
