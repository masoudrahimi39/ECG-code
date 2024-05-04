import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2

def determine_bb_from_saved_image(sample: dict, save_bb: bool):
    """"
    Determine the bounding box; it receives log of a single image and then determine bounding boxes by loading the image.
    Input format: 
        sample : one of the samples from log file. One sample under logs['samples']
    output format :
        List of tuples like: (object_class, x_c, y_c, width, height)
    """
    file_path = sample['save_path']
    file_name = file_path.split("/")[-1].split(".jpg")[0]
    label_path = file_path.replace(file_name + '.jpg', '')

    img = plt.imread(file_path)
    img = img[34:, :-34, :]
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

        # Calculate the coordinates of the box (x, y are centers of the bounding boxes)
        x = int((lead_x_ratio - width_lead_ratio/2) * img_width) 
        y = int((1-(lead_y_ratio + height_lead_ratio/2)) * img_height)
        w = int(width_lead_ratio * img_width) 
        h = int(height_lead_ratio * img_height) 
        
        label.append((0,(x+w/2)/img_width, (y-h/2)/img_height, w/img_width, h/img_height))

    if save_bb == True:
        # change the format of the label in order to write all of the label in one flush rather than writing each line seperately.
        label = '\n'.join(' '.join(str(float_val) for float_val in tup) for tup in label)
        # Write the entire formatted string to the file at once
        with open(f"{label_path+file_name}_copy.txt", "w") as file:
            file.write(label)

    return label

def determine_bb_online(sample: dict, img_array, save_bb_path: str):
    """"
    Determine the bounding box; it received logs of a sample, but the imgae was loaded before calling this function.
    Input format: 
        sample : one of the samples from log file. One sample under logs['samples']
        img_array : the np array of the image
        save_bb_path : the path and name of the text file which will be saved as label.
    output format :
        List of tuples like: (object_class, x_c, y_c, width, height)
    """
    # file_path = sample['save_path']
    # file_name = file_path.split("/")[-1].split(".jpg")[0]
    # label_path = file_path.replace(file_name + '.jpg', '')

    # img = plt.imread(file_path)
    img_array = img_array[34:, :-34, :]   # we have white border in the left and down of the image, so these borders are removed here
    img_size = img_array.shape
    
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

        # Calculate the coordinates of the box (x, y are centers of the bounding boxes)
        x = int((lead_x_ratio - width_lead_ratio/2) * img_width) 
        y = int((1-(lead_y_ratio + height_lead_ratio/2)) * img_height)
        w = int(width_lead_ratio * img_width) 
        h = int(height_lead_ratio * img_height) 
        
        label.append((0,(x+w/2)/img_width, (y-h/2)/img_height, w/img_width, h/img_height))

    if save_bb_path:
        # change the format of the label in order to write all of the label in one flush rather than writing each line seperately.
        label = '\n'.join(' '.join(str(float_val) for float_val in tup) for tup in label)
        # Write the entire formatted string to the file at once
        with open(save_bb_path, "w") as file:
            file.write(label)

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
    img = img[34:, :-34, :]
    img_size = img.shape
    img_width = img_size[1]
    img_height = img_size[0]

    plt.imshow(img)
    for _, x, y, w, h in determine_bb(sample, False):
        lead_width = w*img_width
        lead_height = h*img_height
        plt.gca().add_patch(plt.Rectangle((x*img_width-lead_width/2, y*img_height+lead_height/2),\
                lead_width, lead_height, edgecolor='g', linewidth=0.3, fill=False))
    plt.savefig(export_path, dpi=300)
    plt.close()



if __name__ == "__main__":
    log_path = "datasets/image_dataset_v4.0/train/"
    with open(f"{log_path}logs.json", 'r') as f:
        logs = json.load(f)
    determine_bb_from_saved_image(logs['samples'][13], True)
    draw_bb(logs['samples'][13], "2.jpg")


