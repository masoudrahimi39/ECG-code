import os
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import numpy as np


def match_lead_and_lead_name(parsed_leads, parsed_text):
  for lead_box in parsed_leads:
    distances = []
    x, y, w, h = lead_box['bbox_xywh']

    # Calculate box_lead_upper_left = (x - w/2, y - h/2)
    box_lead_upper_left = np.column_stack((x - w / 2,y - h / 2))

    for lead_text in parsed_texts:
      x_text, y_text, w_text, h_text = lead_text['bbox_xywh']

      # Calculate box_text_lower_left = (x - w/2, y + h/2)
      box_text_lower_left = np.column_stack((x_text - w_text/2, y_text + h_text/2))

      distances.append(np.linalg.norm(box_text_lower_left - box_lead_upper_left, axis=1))

    closet_text_index = np.argmin(distances)

    lead_box['lead_name'] = parsed_texts[closet_text_index]['class_name']

    # todo: we can remove closet_text_index from the parsed_text
    # parsed_text.pop(closet_text_index)
    return parsed_leads


if __name__ == '__main__':
  parsed_text = [{'class_name': 'V2', 'bbox_xywh': (2301.4297, 779.8965, 101.270996, 87.381836)}, {'class_name': 'aVF', 'bbox_xywh': (1326.8567, 1277.6069, 134.29443, 75.7832)}, {'class_name': 'V1', 'bbox_xywh': (2300.647, 286.14166, 99.41284, 92.70076)}, {'class_name': 'II', 'bbox_xywh': (300.82, 780.18744, 59.29718, 86.31348)}, {'class_name': 'II', 'bbox_xywh': (298.7282, 1757.5378, 62.881348, 88.302)}, {'class_name': 'aVR', 'bbox_xywh': (1333.7944, 785.22925, 143.27954, 76.83765)}, {'class_name': 'V3', 'bbox_xywh': (2301.537, 1275.4956, 101.440186, 89.2572)}, {'class_name': 'V5', 'bbox_xywh': (3294.648, 784.295, 96.363525, 73.500244)}, {'class_name': 'V6', 'bbox_xywh': (3294.414, 1279.3103, 98.829834, 70.37988)}, {'class_name': 'III', 'bbox_xywh': (311.98694, 1274.7888, 78.30896, 80.99744)}, {'class_name': 'I', 'bbox_xywh': (294.40582, 284.81848, 46.68036, 93.105255)}]
  parsed_leads = [{'class_name': 'lead', 'bbox_xywh': (2758.1284, 1302.8975, 1007.2693, 403.0984)}, {'class_name': 'lead', 'bbox_xywh': (3754.1172, 1346.9863, 1015.251, 290.932)}, {'class_name': 'lead', 'bbox_xywh': (3746.7507, 803.57825, 1011.7378, 406.79486)}, {'class_name': 'lead', 'bbox_xywh': (1762.297, 529.82196, 1012.02954, 183.80966)}, {'class_name': 'lead', 'bbox_xywh': (775.68005, 904.68414, 1013.5523, 189.00854)}, {'class_name': 'lead', 'bbox_xywh': (2755.5044, 924.1831, 1004.3052, 195.50507)}, {'class_name': 'lead', 'bbox_xywh': (2755.7412, 516.3966, 1004.0608, 287.0313)}, {'class_name': 'lead', 'bbox_xywh': (1762.9132, 1428.0677, 1008.6135, 122.27417)}, {'class_name': 'lead', 'bbox_xywh': (774.22675, 392.99088, 1014.7567, 229.68066)}, {'class_name': 'lead', 'bbox_xywh': (2228.2913, 1879.0068, 4002.6284, 199.7478)}, {'class_name': 'lead', 'bbox_xywh': (3738.6416, 281.9385, 1017.47974, 500.99802)}, {'class_name': 'lead', 'bbox_xywh': (1761.0247, 934.4326, 1010.9042, 118.77057)}, {'class_name': 'lead', 'bbox_xywh': (772.06006, 1457.4872, 1000.3309, 74.930664)}]
  new_parsed_leads = match_lead_and_lead_name(parsed_leads, parsed_text)

  print(f'keys in parsed_text {parsed_text.keys()}')