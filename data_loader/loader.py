# import os
# from dotenv import load_dotenv

# load_dotenv()

# raw_data = os.getenv("npy_path")


# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import matplotlib.pyplot as plt
# import os

# import sys
# sys.path.append("./prepare image dataset/adjusted library")  # in order to import the adjusted "ecg_plot" library
# # Assuming ecg_plot_v2 is a module you have access to that provides the plotting functionality
# import ecg_plot_v2

# class ECGDataset(Dataset):
#     def __init__(self, npy_file, lead_config, transform=None):
#         """
#         Args:
#             npy_file (string): Path to the .npy file with ECG signals.
#             lead_config (dict): Configuration for ECG lead formats.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.X = np.load(npy_file)
#         self.lead_config = lead_config
#         self.transform = transform

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         ecg_signal = self.X[idx]
#         # Generate an image based on the ecg_signal and lead_config here
#         # This is a simplified version, adjust according to your ecg_plot_v2.plot and save_as_jpg functions
#         # For demonstration, let's assume we use a simple plt.plot() and savefig()
        
#         # Select a lead_format from lead_config randomly or in a fixed manner
#         # Here we select '3by1' just for demonstration
#         lead_format = '3by1'
#         each_lead_config = self.lead_config[lead_format]
        
#         fig, ax = plt.subplots()
#         ax.plot(ecg_signal[:each_lead_config['length'], each_lead_config['lead_order'][0]])
#         # Normally, you'd use ecg_plot_v2.plot() here as per your existing code
#         plt.close(fig)
        
#         # Convert plot to image (in memory)
#         fig.canvas.draw()
#         img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
#         if self.transform:
#             img = self.transform(img)
        
#         return img

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(),  # Example augmentation
#     transforms.ToTensor(),
# ])

# lead_config = {'3by1': {'n_column':1, 'length': 1000, 'lead_order': list(range(3)), 'full_ecg_name': None}, 
#                '3by4': {'n_column':4, 'length': 250, 'lead_order': list(range(12)), 'full_ecg_name': 'II'},
#                '12by1':  {'n_column':1, 'length': 1000, 'lead_order': list(range(12)), 'full_ecg_name': None}, 
#                '6by2': {'n_column':2, 'length': 500, 'lead_order': list(range(12)), 'full_ecg_name': 'II'}, 
#                }  # key determines lead format, value determines some variable passing to ecg_plot_vs.plot

# # Define your dataset
# ecg_dataset = ECGDataset(npy_file='all_signals_100Hz.npy', lead_config=lead_config, transform=transform)

# # Create DataLoader
# ecg_dataloader = DataLoader(ecg_dataset, batch_size=4, shuffle=True)


import json
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

# Create a figure and axis

# Open the JSON file
with open('logs.json', 'r') as f:
    # Load the JSON data
    data = json.load(f)

d = data[0]

# Load the image
img = Image.open(f"./{d['file_name']}.jpg")

# Plot the image
plt.close('all')
plt.imshow(img)
plt.axis('off')  # Turn off axis
plt.savefig("o.png")
# plt.show()


plt.close('all')
img = plt.imread(f"./{d['file_name']}.jpg")
plt.imshow(img)
plt.savefig("o.png")

height, width, layers = img.shape

y_min, y_max, x_min, x_max = d['y_min'], d['y_max'], d['x_min'], d['x_max']
log_height, log_width = y_max - y_min, x_max - x_min

plt.close('all')
fig, ax = plt.subplots()
ax.imshow(img)
for l in d['leads']:
    min_x_plot, max_x_plot, min_y_plot, max_y_plot = l['min_x_plot'], l['max_x_plot'], l['min_y_plot'], l['max_y_plot']

    lead_start_x = (min_x_plot-x_min)/log_width * width
    lead_end_x = (1-(x_max-max_x_plot)/log_width) * width

    lead_start_y = (1-(min_y_plot-y_min)/log_height) * height
    lead_end_y = ((y_max-max_y_plot)/log_height) * height

    rec = patches.Rectangle((lead_start_x, lead_start_y), lead_end_x-lead_start_x, lead_end_y-lead_start_y, linewidth=0.4, edgecolor='red', facecolor='none')
    ax.add_patch(rec)
plt.savefig("o.png", dpi=200)
