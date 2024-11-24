#!$HOME/.pyenv/shims/python
# coding: utf-8

import numpy as np
import os
from pathlib import Path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level to project root
if project_root not in sys.path:
    sys.path.append(project_root)
    print(project_root)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# Standardize the path for Windows
project_root = os.path.normpath(project_root)
os.environ['PYTHONPATH'] = project_root


if 'PYTHONPATH' not in os.environ:
    print("Error: Run `source env.sh` to be able to run `/scripts/*.py`")
    exit(1)

import ecgdigitize
from ecgdigitize import visualization
from ecgdigitize.image import BinaryImage, ColorImage, openImage, saveImage
from ecgdigitize.signal import detection as signal_detection
from ecgdigitize.signal.extraction import viterbi


DATASET_PATH


def validate(DATASET_PATH):
    for path in DATASET_PATH:
        image = openImage(Path(path))
        # (1) Color -> Binary
        binary = signal_detection.adaptive(image)
        # (2) Binary -> Signal
        signal = viterbi.extractSignal(binary)
        
        if signal is not None:
            output = visualization.overlaySignalOnImage(signal, binary.toColor())
        else:
            output = binary.toColor()

        outputPath = f"validation/signal/{Path(path).name}"
        saveImage(output, Path(outputPath))
        print(f"Saved '{outputPath}'")


def testSingle(path: str):
    image = openImage(Path(path))

    binary = signal_detection.adaptive(image)
    signal = ecgdigitize.digitizeSignal(binary)

    visualization.displayImage(binary.toColor())

    if signal is not None:
        output = visualization.overlaySignalOnImage(signal, binary.toColor())
    else:
        output = binary.toColor()
    print('output type: ', type(output))
    print('output type: ', len(output))
    visualization.displayImage(output)
    # outputPath = f'/media/masoud/F45C0A0B5C09C976/Education/4_ECG_digitization/ECG-code/digi_out/{path.split("/")[-1]}'
    outputPath = r"D:\Education\4_ECG_digitization\ECG-code\digi_out"

    np.save(outputPath, signal)
    saveImage(output, Path(outputPath))
    print(f"Saved {outputPath}")


if __name__ == "__main__":
    # validate()
    # testSingle('lead-pictures/040-46000898.png')
    # path = '/media/masoud/F45C0A0B5C09C976/Education/4_ECG_digitization/ECG-code/datasets/image_dataset_v6.0/digitization/train/images/0_3by1_lead_1.png'
    path = r"D:\Education\4_ECG_digitization\ECG-code\datasets\image_dataset_v6.0\digitization\train\images\0_3by1_lead_1.png"

    get_signal_from_binary(path)

# TODO:
# - Treat all pixels as nodes? -- probably too slow
# - Add prior weights that incentivize nodes that are in the middle (horizontall and vertically) of other pixels to draw the line to the center of the signal trace