from dotenv import load_dotenv
from ultralytics import YOLO
import json
import os
import numpy as np
import cv2

load_dotenv()
model_path = os.getenv("segmentation_model")
segmentation_model = YOLO(model_path)

def get_masked(image):
    # image is an np array shaped signal (height, width, channels) or any thing that could be converted to such an array with this command: np.array(image)
    # output is a masked np array (only 0 and 1)
    # sometimes, network can't process well and all the array will be zero !
    im = np.array(image)
    result = segmentation_model(im)
    if isinstance(result[0].masks, type(None)):
        print("\033[31mERROR IN SEGMENTATION\033[0m")
        return np.zeros((im.shape[0], im.shape[1]))
    return result[0].masks[0].data[0]

def erode_image(input_image):
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(np.uint8(input_image), kernel, iterations=1)
    return img_erosion

if __name__ == "__main__":
    from prepare_image_dataset.utils import crop_bb
    import matplotlib.pyplot as plt

    log_path = os.path.join(os.getenv("detection_dataset_path"), "train", "logs.json")    

    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    sample = logs['samples'][2]
    cropped_images = crop_bb(sample=sample, export_path=False, img_path=sample['save_path'], prefix="")

    for cropped_image in cropped_images:
        masked = get_masked(cropped_image)
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

