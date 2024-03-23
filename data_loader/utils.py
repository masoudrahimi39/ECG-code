import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os
import json

def return_bounding_boxes(img, log):
    height, width, layers = img.shape
    y_min, y_max, x_min, x_max = log['y_min'], log['y_max'], log['x_min'], log['x_max']
    log_height, log_width = y_max - y_min, x_max - x_min

    bounding_boxes = []

    for l in log['leads']:
        min_x_plot, max_x_plot, min_y_plot, max_y_plot = l['min_x_plot'], l['max_x_plot'], l['min_y_plot'], l['max_y_plot']

        lead_start_x = (min_x_plot-x_min)/log_width * width
        lead_end_x = (1-(x_max-max_x_plot)/log_width) * width

        lead_start_y = (1-(min_y_plot-y_min)/log_height) * height
        lead_end_y = ((y_max-max_y_plot)/log_height) * height

        lead_width = lead_end_x-lead_start_x
        lead_height = lead_end_y-lead_start_y

        bounding_boxes.append([lead_start_x, lead_start_y, lead_width, lead_height])
    
    return bounding_boxes

def plot_bounding_boxes(img, log, save_path):
    plt.close('all')
    bounding_boxes = return_bounding_boxes(img, log)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for bb in bounding_boxes:
        rec = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=0.4, edgecolor='red', facecolor='none')
        ax.add_patch(rec)
    plt.savefig(save_path, dpi=200)

if __name__ == '__main__':
    path = "datasets/image_dataset_v4.0/train/"
    with open(f"{path}logs.json", 'r') as f:
        logs = json.load(f)['samples']
    log = logs[0]
    img = plt.imread(f"{path}{log['image_name']}.jpg")
    plot_bounding_boxes(img, log, "./tmp.jpg")    