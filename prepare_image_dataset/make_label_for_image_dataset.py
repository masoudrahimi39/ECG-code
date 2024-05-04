
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from utils import determine_bb_from_saved_image
load_dotenv()

dataset_version = 4 
img_dataset_dir = f'{os.getenv("datasets_path")}image_dataset_v{dataset_version}.0/'
# Iterate over each directory (train, test, val) in the image_datest directory
for subdir in os.listdir(img_dataset_dir):
    subdir_path = os.path.join(img_dataset_dir, subdir)
    # Check if the current item in the main directory is a directory itself
    if os.path.isdir(subdir_path):
        # Construct the path to logs.json in the current subdirectory
        logs_json_path = os.path.join(subdir_path, 'logs.json')
        # Check if logs.json exists in the current subdirectory
        if os.path.exists(logs_json_path):
            # Load and process logs.json
            with open(logs_json_path, 'r') as f:
                logs_data = json.load(f)
                # iterate over samples in the log file (each sample is )
                for img in logs_data['samples']:
                    # save a text file containing labels
                    determine_bb_from_saved_image(sample=img, save_bb=True)
        else:
            print('logs.json file is not available')

