import pandas as pd
import numpy as np
import wfdb
import ast
import json
import matplotlib.pyplot as plt
import os
import sys
from math import ceil
import shutil
sys.path.append("./prepare_image_dataset/adjusted_library")  # in order to import the adjusted "ecg_plot" library
import ecg_plot_v2
from dotenv import load_dotenv
load_dotenv()

# TODO: look after the frequency
sampling_rate = 100

# if the .npy file of signal is not available, we will make it; otherwise, it will be loaded
if not os.path.exists(f'{os.getenv("signal_dataset_path")}all_signals_{sampling_rate}Hz.npy'):  
    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    path = os.getenv("raw_signal_path")

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    np.save(f'{os.getenv("signal_dataset_path")}all_signals_{sampling_rate}Hz.npy', X)
    print(f'signal dataset was saved into all_signals_{sampling_rate}Hz.npy file')
else:
    X = np.load(f'{os.getenv("signal_dataset_path")}all_signals_{sampling_rate}Hz.npy')
    print(f'all_signals_{sampling_rate}Hz.npy file loaded')


### convert the signal to ecg image
dataset_version = 4 
n_sample_each_lead = 17       # number of images with same lead format; if n_sample_each_lead == 500, first 500 signals will be printed as 3by1 lead format, the next 500 signals will be printed as 3by4 format
# percent of train, validation, and test sets
train_prcnt = 0.6
val_prcnt = 0.2
test_prcnt = 0.2

path_to_save_dataset = f'{os.getenv("datasets_path")}image_dataset_v{dataset_version}.0/'
if os.path.exists(path_to_save_dataset):
    shutil.rmtree(path_to_save_dataset)   # remove the directory

os.makedirs(path_to_save_dataset)
# Create child directories: test, train, val
os.makedirs(os.path.join(path_to_save_dataset, "test"))
os.makedirs(os.path.join(path_to_save_dataset, "train"))
os.makedirs(os.path.join(path_to_save_dataset, "val"))
print(f'directory {path_to_save_dataset} created.')


row_height = 6.265
lead_index = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # order of the leads in the dataset
lead_display = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # order of the lead that I want to be shown
lead_config = {'3by1': {'n_column':1, 'length': 1000, 'lead_order': list(range(3)), 'full_ecg_name': None, 'n_leads': 3}, 
               '3by4': {'n_column':4, 'length': 250, 'lead_order': [0,1,2,4,3,5,6,7,8,9,10,11], 'full_ecg_name': 'II', 'n_leads': 12},
               '12by1':  {'n_column':1, 'length': 1000, 'lead_order': [0,1,2,4,3,5,6,7,8,9,10,11], 'full_ecg_name': None, 'n_leads': 12}, 
               '6by2': {'n_column':2, 'length': 500, 'lead_order': [0,1,2,4,3,5,6,7,8,9,10,11], 'full_ecg_name': 'II', 'n_leads': 12}, 
               }  # key determines lead format, value determines some variable passing to ecg_plot_vs.plot

cnt = 0   # it is used to handle the number of images for each lead
logs = {'train': [], 'test': [], 'val': []}
#TODO: now, some values are fixed but we should choose them randomly. (records for train, validation, test - records for each lead)
for lead_format, each_lead_config in lead_config.items():
    cnt2 = 0  # to track number of samples in each set (train, validation, test)
    for i in range(cnt, cnt + n_sample_each_lead):    # using cnt and n_sample_each_lead, we will have first 500 (`step=500`) data in the first lead_format and the next 500 data will be in the second lead_format and so on 
        
        # check to put the data in which set (test, train, val)
        if cnt2 < ceil(train_prcnt*n_sample_each_lead):
            dtset_type = 'train'
        elif ceil(train_prcnt*n_sample_each_lead) <= cnt2 < ceil((val_prcnt+train_prcnt)*n_sample_each_lead) :
            dtset_type = 'val'
        elif ceil((val_prcnt+train_prcnt)*n_sample_each_lead) <= cnt2 :
            dtset_type = 'test'
        
        logs[dtset_type].append(
            ecg_plot_v2.plot(
                ecg=X[i, :each_lead_config['length'], :each_lead_config['n_leads']].T,
                full_ecg=X[i, :, 1].T, 
                full_ecg_name=each_lead_config['full_ecg_name'],
                sample_rate=sampling_rate, 
                columns=each_lead_config['n_column'],
                lead_index=lead_index,
                title='', 
                lead_order=each_lead_config['lead_order'],
                show_lead_name=True,
                show_grid=True, 
                show_separate_line=True,
                row_height=row_height,
                style=None)
        )
        file_name = str(i)+'_'+lead_format
        logs[dtset_type][-1]["image_name"] = file_name

        ecg_plot_v2.save_as_jpg(file_name, path=os.path.join(path_to_save_dataset, dtset_type + '/'), dpi=100)
        plt.close()
        cnt2 +=1

    cnt += n_sample_each_lead

for data_set_type, value in logs.items():
    with open(f"{os.path.join(path_to_save_dataset, data_set_type + '/')}logs.json", 'w') as f:
        json.dump({"frequency": sampling_rate, "samples":value}, f)