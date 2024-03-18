import pandas as pd
import numpy as np
import wfdb
import ast
import json
import matplotlib.pyplot as plt
import os
import sys
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
n_sample_each_lead = 5       # number of images with same lead format; if step == 500, first 500 signals will be printed as 3by1 lead format, the next 500 signals will be printed as 3by4 format
dataset_version = 4
path_to_save_dataset = f'{os.getenv("datasets_path")}image_dataset_v{dataset_version}.0/'
if not os.path.exists(path_to_save_dataset):
    os.makedirs(path_to_save_dataset)
    print(f'directory {path_to_save_dataset} created.')

row_height = 6.265

#TODO: check lead_display in the generated file
lead_index = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # order of the leads in the dataset
lead_display = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # order of the lead that I want to be shown
lead_config = {'3by1': {'n_column':1, 'length': 1000, 'lead_order': list(range(3)), 'full_ecg_name': None, 'n_leads': 3}, 
               '3by4': {'n_column':4, 'length': 250, 'lead_order': [0,1,2,4,3,5,6,7,8,9,10,11], 'full_ecg_name': 'II', 'n_leads': 12},
               '12by1':  {'n_column':1, 'length': 1000, 'lead_order': [0,1,2,4,3,5,6,7,8,9,10,11], 'full_ecg_name': None, 'n_leads': 12}, 
               '6by2': {'n_column':2, 'length': 500, 'lead_order': [0,1,2,4,3,5,6,7,8,9,10,11], 'full_ecg_name': 'II', 'n_leads': 12}, 
               }  # key determines lead format, value determines some variable passing to ecg_plot_vs.plot
cnt = 0
logs = []
for lead_format, each_lead_config in lead_config.items():
    for i in range(cnt, cnt + n_sample_each_lead):    # using cnt and n_sample_each_lead, we will have first 500 (`step=500`) data in the first lead_format and the next 500 data will be in the second lead_format and so on 
        logs.append(
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
        logs[-1]["image_name"] = file_name
        ecg_plot_v2.save_as_jpg(file_name, path=path_to_save_dataset, dpi=100)
        plt.close()


    cnt += n_sample_each_lead

with open(f'{path_to_save_dataset}logs.json', 'w') as f:
    json.dump({"frequency": sampling_rate, "samples":logs}, f)

###