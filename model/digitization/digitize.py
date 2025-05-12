#!$HOME/.pyenv/shims/python
# coding: utf-8

from scipy.signal import correlate
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import numpy as np
import os
from pathlib import Path
import sys
import json
import cv2
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
if 'PYTHONPATH' not in os.environ:
    os.environ['PYTHONPATH'] = './'
else:
    os.environ['PYTHONPATH'] += os.pathsep + './'

if 'PYTHONPATH' not in os.environ:
    print("Error: Run `source env.sh` to be able to run `/scripts/*.py`")
    exit(1)

from viterbi import extractSignal
from signal_detection import adaptive
from image import ColorImage, GrayscaleImage, BinaryImage, openImage, saveImage
import common
import ecgdigitize
#######################################
# visualization
########################################

def overlaySignalOnImage(
    signal: np.ndarray,
    image: ColorImage,
    color: Tuple[np.uint8, np.uint8, np.uint8] = (85, 19, 248),
    lineWidth: int = 1
) -> ColorImage:
    assert len(signal.shape) == 1
    assert isinstance(image, ColorImage)

    def quantize(element: float) -> Optional[int]:
        if not np.isnan(element):
            return int(element)
        else:
            return None

    output = image.data.copy()
    quantizedSignal = common.mapList(signal, quantize)

    for first, second in zip(enumerate(quantizedSignal[:-1]), enumerate(quantizedSignal[1:], start=1)):
        if first[1] is not None and second[1] is not None:
            cv2.line(output, first, second, color, thickness=lineWidth)
    print('type', type(output))
    print(output.shape, np.unique(output))
    return ColorImage(output)

#######################################
# 
########################################


def process_digitized_signal(digitized_signal, real_signal, baseline=0.005, 
                             y_scale=0.0064, x_scale=2.54
                            # y_scale=0.01802, x_scale=7.16332378                             
                            ):
    '''Process and align the extracted signal with the real signal.'''
    # Remove NaN values
    digitized_signal = digitized_signal[~np.isnan(digitized_signal)]
    
    # scale the signal based on the pixel size
    x_vals = np.arange(len(digitized_signal))
    digitized_signal, x_vals = digitized_signal*y_scale, x_vals*x_scale

    # interpolate the signal to match the real signal length
    original_indices = np.linspace(0, 1, len(digitized_signal))
    real_signal_len = len(real_signal)
    new_indices = np.linspace(0, 1, real_signal_len)  # Interpolating to 1000 points
    interpolation_function = interp1d(original_indices, digitized_signal, kind='linear')
    y_vals = interpolation_function(new_indices)

    # shift y values of the signal
    y_vals = -(y_vals - np.median(y_vals) + baseline)
    
    # handle time misalignment (shift the signal in x axis)
    lag = np.argmax(correlate(real_signal, y_vals)) - (len(y_vals) - 1)
    y_vals = np.roll(y_vals, lag)
    #TODO: maybe lag is constant for all images so no need for taking real_signal as an argument
    #TODO: remove below print statement
    print(f'lag: {lag}')
    x_vals = x_vals - np.min(x_vals)
    
    return x_vals, y_vals

def get_image_paths(dataset_path):
    """Returns list of .png files in the given directory."""
    return list(Path(dataset_path).glob('*.png'))



def get_final_signal(*, label_path: str, output_path: str, save_image: bool, image_path=None, image=None):
    '''get path of a single image and return the digitized signal and its correlation with the real signal'''
    if image_path:
        # check if we have .png or .bmp image_path
        if image_path.split('.')[-1] == 'png':
            image_path = Path(image_path)
            image = openImage(image_path, png=True)
            binary = adaptive(image)
            signal = ecgdigitize.digitizeSignal(binary)
            
        elif image_path.split('.')[-1] == 'bmp':
            image_path = Path(image_path)
            binary = openImage(image_path, bmp=True)
            # binary = adaptive(binary)
            signal = ecgdigitize.digitizeSignal(binary)
            # signal = extractSignal(binary)


    elif image:
        #todo: we need to accept image (np nd array) as input
        pass



    with open(label_path, 'r') as f:
        real_signal = json.load(f)

    # try:
    if save_image:
        output = overlaySignalOnImage(signal, binary.toColor())
        output_image_path = os.path.join(output_path, Path(image_path).name)
        saveImage(output, Path(output_image_path))
    
    # x an y values of the digitized signal after processing
    # x_vals, y_vals = process_digitized_signal(signal, real_signal)
    print('signal shape:', signal.__len__())
    print('readl signal shape:', real_signal.__len__())
    # correlation = pearsonr(real_signal, y_vals)[0]
    # print(f'corr: {correlation}')
    # print(f'MSE: {np.mean(abs(y_vals - real_signal))}')
    plt.plot(signal, 'r')
    
    plt.plot(real_signal, 'b')
    plt.show()

    filename = "digitized_lead_1_code.txt"  # File to save the values

# Open the file in write mode and write each element on a new line
    with open(filename, "w") as file:
        for value in signal:
            file.write(f"{value}\n")

    # return correlation
    # except Exception as e:
    #     print(f"Error processing {image_path}: {str(e)}")
    #     return None

def validate(DATASET_PATH, LABELS_PATH, OUTPUT_PATH, save_image: bool):
    '''get paths to input image and correpondig labels and return the average correlation and the digitized signals'''
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    correlations = []
    digitized_signals = []
    for image_path in get_image_paths(DATASET_PATH):
        label_path = os.path.join(LABELS_PATH, f"{Path(image_path).stem}.json") # Debug print
        print(20*'--')
        print(f"Processing image: {image_path}") 
        print(f"label_path: {label_path}")
        x_vals, y_vals, correlation = get_final_signal(image_path=image_path, label_path=label_path, output_path=OUTPUT_PATH, save_image=save_image) 
        print(f"correlation: {correlation}")
        correlations.append(correlation)
        # digitized_signals.append(y_vals)

    print(f"Average correlation: {np.mean(correlations)}")


# def testSingle(path: str):
#     image = openImage(Path(path))

#     binary = signal_detection.adaptive(image)
#     signal = ecgdigitize.digitizeSignal(binary)

#     visualization.displayImage(binary.toColor())

#     if signal is not None:
#         output = visualization.overlaySignalOnImage(signal, binary.toColor())
#     else:
#         output = binary.toColor()
#     print('output type: ', type(output))
#     print('output type: ', len(output))
#     visualization.displayImage(output)


if __name__ == "__main__":
    # DATASET_PATH = r'datasets/image_dataset_v6.0/digitization/train/images'
    # LABELS_PATH = r'datasets/image_dataset_v6.0/digitization/train/labels'
    # OUTPUT_PATH = r'model/digitization/results'
    # validate(DATASET_PATH, LABELS_PATH, OUTPUT_PATH, save_image=True)

    get_final_signal(
                    # image_path=r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\segmentation\train\labels\0_3by1_lead_0.bmp',
                    #  image_path=r'datasets\image_dataset_v7.0\digitization\train\images\0_3by1_lead_0.png',
                    #  label_path=r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\digitization\train\labels\0_3by1_lead_0.json',
                    #  output_path=r'D:\Education\4_ECG_digitization\ECG-code', 
                    #  save_image=True)
                    image_path=r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\segmentation\train\labels\0_3by1_lead_0.bmp',
                    # image_path=r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v6.0\digitization\train\images\0_3by1_lead_0.png',
                     label_path=r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\digitization\train\labels\0_3by1_lead_0.json',
                     output_path=r'D:\Education\4_ECG_digitization\ECG-code', 
                     save_image=True)
    

    # get_final_signal(
    #                 # image_path=r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\segmentation\train\labels\0_3by4_lead_0.bmp',
    #                  image_path=r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\digitization\train\images\0_3by1_lead_0.png',
    #                  label_path=r'D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v7.0\digitization\train\labels\0_3by1_lead_0.json',
    #                  output_path=r'D:\Education\4_ECG_digitization\ECG-code', 
    #                  save_image=True)

# # TODO:
# # - Treat all pixels as nodes? -- probably too slow
# # - Add prior weights that incentivize nodes that are in the middle (horizontall and vertically) of other pixels to draw the line to the center of the signal trace