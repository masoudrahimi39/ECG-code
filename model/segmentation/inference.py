from dotenv import load_dotenv
from ultralytics import YOLO
import json
import os
import numpy as np
import cv2

def get_masked(model, image):
    result = segmentation_model(image)
    if isinstance(result[0].masks, type(None)):
        print("\033[31mERROR IN SEGMENTATION\033[0m")
        return None
    return result[0].masks[0].data[0]

def erode_image(input_image):
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(np.uint8(input_image), kernel, iterations=1)
    return img_erosion

if __name__ == "__main__":
    from prepare_image_dataset.utils import crop_bb
    import matplotlib.pyplot as plt

    load_dotenv()
    model_path = os.getenv("segmentation_model")
    log_path = os.path.join(os.getenv("detection_dataset_path"), "train", "logs.json")
    
    segmentation_model = YOLO(model_path)

    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    sample = logs['samples'][2]
    cropped_images = crop_bb(sample=sample, export_path=False, img_path=sample['save_path'], prefix="")

    for cropped_image in cropped_images:
        masked = get_masked(segmentation_model, cropped_image)
        if isinstance(masked, type(None)):
            continue
        erroded = erode_image(masked)

        plt.imshow(cropped_image)
        plt.savefig("test_segmentation_input.png")
        plt.close()

        plt.imshow(masked)
        plt.savefig("test_segmentation_model.png")
        plt.close()

        plt.imshow(erroded)
        plt.savefig("test_segmentation_erroded.png")
        plt.close()

