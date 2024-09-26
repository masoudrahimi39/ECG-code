
# apply easyocr on text images

import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from dotenv import load_dotenv
import os

def load_easy_ocr_model():
    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    return reader

def apply_easyocr(*, cropped, reader=None, plot=False, save=False, save_dir=None):
    cropped_img = cv2.cvtColor(np.array(cropped), cv2.COLOR_BGR2GRAY)
    cropped_img[cropped_img>40] = 255

    reader_result = reader.recognize(np.array(cropped_img), allowlist=['a', 'F', 'R', 'V', 'I', 'l', 'L', '|', '1', '2', '3', '4', '5', '6'])

    out = [r[1] for r in reader_result]
    out = "".join(out)
    out = out.replace("l", "I").replace("|", "I")

    if plot:
        plt.title(f"EasyOCR prediction: {out}")
        plt.imshow(cropped_img)
        plt.show()
        plt.close("all")
    if save:
        plt.title(f"EasyOCR prediction: {out}")
        plt.imshow(cropped_img)
        plt.savefig(f'{save_dir}/{i}.jpg')
        plt.close("all")
    return out


if __name__ == '__main__':
    load_dotenv()
    detection_classes = os.getenv("labels_dict")
    classes = json.loads(detection_classes)
    classes = {v:k for k, v in classes.items()}

    # todo: parsed_texts should be update. it does not contain 'img'. after updating, check this function
    parsed_texts = [{'class_name': 'V2', 'bbox_xywh': (2301.4297, 779.8965, 101.270996, 87.381836)}, {'class_name': 'aVF', 'bbox_xywh': (1326.8567, 1277.6069, 134.29443, 75.7832)}, {'class_name': 'V1', 'bbox_xywh': (2300.647, 286.14166, 99.41284, 92.70076)}, {'class_name': 'II', 'bbox_xywh': (300.82, 780.18744, 59.29718, 86.31348)}, {'class_name': 'II', 'bbox_xywh': (298.7282, 1757.5378, 62.881348, 88.302)}, {'class_name': 'aVR', 'bbox_xywh': (1333.7944, 785.22925, 143.27954, 76.83765)}, {'class_name': 'V3', 'bbox_xywh': (2301.537, 1275.4956, 101.440186, 89.2572)}, {'class_name': 'V5', 'bbox_xywh': (3294.648, 784.295, 96.363525, 73.500244)}, {'class_name': 'V6', 'bbox_xywh': (3294.414, 1279.3103, 98.829834, 70.37988)}, {'class_name': 'III', 'bbox_xywh': (311.98694, 1274.7888, 78.30896, 80.99744)}, {'class_name': 'I', 'bbox_xywh': (294.40582, 284.81848, 46.68036, 93.105255)}]
    reader = load_easy_ocr_model()
    ocred_texts = []
    for p in parsed_texts:
        pred = apply_easyocr(cropped=p['img'], reader=reader, save=True, 
        save_dir=f'./model/detection/inf_img/{10}.jpg')

        ocred_texts.append(p.copy())
        ocred_texts[-1]['ocr-pred'] = pred

    print(ocred_texts)