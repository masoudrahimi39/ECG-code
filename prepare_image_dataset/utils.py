import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2

def determine_bb(sample: dict):
    """"
    Determine the bounding box for a given sample.
    Input format: 
        sample : one of the samples from log file. One sample under logs['samples']
    output format :
        List of tuples like: (object_class, x_c, y_c, width, height)
    """
    file_path = sample['save_path']
    file_name = file_path.split("/")[-1].split(".jpg")[0]

    img = plt.imread(file_path)
    img_size = img.shape
    
    img_width = img_size[1]
    img_height = img_size[0]

    x_min = sample['x_min']
    x_max = sample['x_max']
    y_min = sample['y_min']
    y_max = sample['y_max']

    width = x_max - x_min
    height = y_max - y_min

    label = []

    for lead in sample['leads']:
        min_x_plot = lead['min_x_plot']
        max_x_plot = lead['max_x_plot']
        min_y_plot = lead['min_y_plot']
        max_y_plot = lead['max_y_plot']

        lead_x_text = lead['x_text']
        lead_y_text = lead['y_text']
        lead_fontsize = lead['fontsize']
        lead_text = lead['text']
        
        width_lead = max_x_plot - min_x_plot
        height_lead = max_y_plot - min_y_plot
        
        lead_x_ratio = ((min_x_plot+max_x_plot)/2-x_min)/width
        lead_y_ratio = ((min_y_plot+max_y_plot)/2-y_min)/height

        width_lead_ratio = width_lead/width
        height_lead_ratio = height_lead/height

        # Calculate the coordinates of the box
        x = int((lead_x_ratio - width_lead_ratio/2) * img_width) + 5
        y = int((1-(lead_y_ratio + height_lead_ratio/2)) * img_height) - 20
        w = int(width_lead_ratio * img_width) + 30
        h = int(height_lead_ratio * img_height) + 30
        
        label.append((0,(x+w/2)/img_width,(y-h/2)/img_height,w/img_width,h/img_height))
    
    return label

def draw_bb(sample: dict, export_path: str):
    """
    Draws a green bounding box around the leads of the image.
    Input format:
        sample: the dictionary which includes the information of the specified sample
        export_path: the path to export the output image
    output:
        an annotated image to the export path
    """
    file_path = sample['save_path']

    img = plt.imread(file_path)
    img_size = img.shape
    img_width = img_size[1]
    img_height = img_size[0]

    plt.imshow(img)
    for _, x, y, w, h in determine_bb(sample):
        lead_width = w*img_width
        lead_height = h*img_height
        plt.gca().add_patch(plt.Rectangle((x*img_width-lead_width/2, y*img_height+lead_height/2),\
                lead_width, lead_height, edgecolor='g', linewidth=1, fill=False))
    plt.savefig(export_path, dpi=300)
    plt.close()


# def save_logs(logs, export_path)

if __name__ == "__main__":
    log_path = "datasets/image_dataset_v4.0/train/"
    with open(f"{log_path}logs.json", 'r') as f:
        logs = json.load(f)
    determine_bb(logs['samples'][0])
    draw_bb(logs['samples'][0], "2.jpg")

