#%%
import pandas as pd
import numpy as np
import wfdb
import ast
import random
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("./prepare image dataset/adjusted library")  # in order to import the adjusted "ecg_plot" library
import ecg_plot_v2
from dotenv import load_dotenv
import json

load_dotenv()

#%% ### load .npy file containing signals
sampling_rate = 100
X = np.load(os.getenv("npy_path"))
lead_index = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # order of the leads in the dataset
lead_display = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # order of the lead that I want to be shown


#%% ### save image datset in drive
# TODO: image scale should be corrected(masoud will do it)
# TODO: add label for object detection task (i.e., bounding box, etc) (reza will do it)

lead_config = {
                # '3by1': {'n_column':1, 'length': 1000, 'lead_order': list(range(3)), 'full_ecg_name': None}, 
               '3by4': {'n_column':4, 'length': 250, 'lead_order': list(range(12)), 'full_ecg_name': 'II'},
            #    '12by1':  {'n_column':1, 'length': 1000, 'lead_order': list(range(12)), 'full_ecg_name': None}, 
            #    '6by2': {'n_column':2, 'length': 500, 'lead_order': list(range(12)), 'full_ecg_name': 'II'}, 
               }  # key determines lead format, value determines some variable passing to ecg_plot_vs.plot
cnt = 0
step = 2       # number of images with same lead format; if step == 500, first 500 signals will be printed as 3by1 lead format, the next 500 signals will be printed as 3by4 format
num_pictures = 0
logs = []

# TODO: add logs of the file and sample to the logs!
for lead_format, each_lead_config in lead_config.items():
    for i in range(cnt, cnt + step):    # using cnt and step, we will have first 500 (`step=500`) data in the first lead_format and the next 500 data will be in the second lead_format and so on 
        logs.append(ecg_plot_v2.plot(
            ecg=X[i, :each_lead_config['length'], :].T,
            full_ecg=X[i, :, 1].T,
            full_ecg_name=each_lead_config['full_ecg_name'],
            sample_rate=100,
            columns=each_lead_config['n_column'],
            lead_index=lead_index,
            title='',
            lead_order=each_lead_config['lead_order'],
            show_lead_name=True,
            show_grid=True,
            show_separate_line=True,
            row_height=6.3,
            style=None,
        ))
        file_name = str(i)+'_'+lead_format
        logs[-1]["file_name"] = file_name
        ecg_plot_v2.save_as_jpg(file_name, path="./", dpi=100)
        plt.close()

    cnt += step

with open(f'./logs.json', 'w') as f:
    json.dump(logs, f)

