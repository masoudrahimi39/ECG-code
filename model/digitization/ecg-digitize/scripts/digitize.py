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
import PIL

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

import ecgdigitize
from ecgdigitize import visualization
from ecgdigitize.image import BinaryImage, ColorImage, openImage, saveImage
from ecgdigitize.signal import detection as signal_detection
from ecgdigitize.signal.extraction import viterbi

import dataclasses
# import PIL
from PIL import Image as pilimage
import cv2
from typing import Optional, Tuple, Union

@dataclasses.dataclass(frozen=True)
class Image:
    data: np.ndarray

    @property
    def height(self):
        return self.data.shape[0]

    @property
    def width(self):
        return self.data.shape[1]


class ColorImage(Image):

    def __post_init__(self) -> None:
        assert isinstance(self.data, np.ndarray)
        assert len(self.data.shape) == 3 and self.data.shape[2] == 3

    def toGrayscale(self):  # -> GrayscaleImage
        """Uses:
            `grey = 0.299 * red + 0.587 * green, 0.114 * blue`

        The "standard method" given in equation (2) by Mallawaarachchi.
        """
        return GrayscaleImage(
            cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        )


class GrayscaleImage(Image):

    def __post_init__(self) -> None:
        assert isinstance(self.data, np.ndarray)
        assert len(self.data.shape) == 2

    def toColor(self) -> ColorImage:
        return ColorImage(cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR))

    def toBinary(self, threshold: Optional[int] = None, inverse: bool =True):  # -> BinaryImage
        if threshold is None:
            if inverse:
                binaryData: np.ndarray
                _, binaryData = cv2.threshold(self.data, 0, 1, cv2.THRESH_OTSU)
                # Convert from uint8 -> bool, invert truth values, then convert bool -> uint8
                binaryData = np.invert(binaryData.astype('bool')).astype('uint8')
            else:
                _, binaryData = cv2.threshold(self.data, 0, 1, cv2.THRESH_OTSU)
        else:
            if inverse:
                _, binaryData = cv2.threshold(self.data, threshold, 1, cv2.THRESH_BINARY_INV)
            else:
                _, binaryData = cv2.threshold(self.data, threshold, 1, cv2.THRESH_BINARY)

        return BinaryImage(binaryData)

    def normalized(self):  # -> GrayscaleImage:
        # Maps the image to the range [0,1]
        assert self.data.dtype is np.dtype('uint8')
        return GrayscaleImage(self.data / 255)

    def whitePointAdjusted(self, strength: float = 1.0):  # -> GrayscaleImage:
        hist = self.histogram()
        whitePoint = np.argmax(hist)
        whiteScaleFactor = 255 / whitePoint * strength
        return GrayscaleImage(cv2.addWeighted(self.data, whiteScaleFactor, self.data, 0, 0))

    def histogram(self) -> np.ndarray:
        counts, _ = np.histogram(self.data, 255, range=(0,255))
        return counts


class BinaryImage(Image):

    def __post_init__(self) -> None:
        assert isinstance(self.data, np.ndarray)
        assert len(self.data.shape) == 2

    def toColor(self) -> ColorImage:
        return ColorImage(cv2.cvtColor(self.data * 255, cv2.COLOR_GRAY2BGR))

    def toGrayscale(self) -> GrayscaleImage:
        return GrayscaleImage(self.data * 255)


def openImage(path: Path, bmp=None, png=None) -> ColorImage:
    assert isinstance(path, Path)
    assert path.exists()

    if png:
        data = cv2.imread(str(path))    
        assert data is not None
        return ColorImage(data)
    
    elif bmp:
        data = pilimage.open(path).convert('L')  # load single channel
        data = np.array(data) 
        assert data is not None
        return BinaryImage(data)


def saveImage(image: Image, path: Path) -> None:
    assert isinstance(image, (ColorImage, GrayscaleImage, BinaryImage))

    if isinstance(image, ColorImage):
        outputImage = image
    elif isinstance(image, GrayscaleImage):
        outputImage = image.toColor()
    elif isinstance(image, BinaryImage):
        outputImage = image.toColor()
    else:
        raise AssertionError

    return cv2.imwrite(str(path), outputImage.data)


#####################################
# viterbi
#####################################
"""
viterbi.py
Created June 1, 2021

...
"""

from math import isnan, sqrt, asin, pi
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple, TypeVar, Union
import numpy as np
import matplotlib.pyplot as plt

import common
from dataclasses import dataclass
# Types
# List = list
Int = int
Float = float
Numeric = Union[Float, Int]

A = TypeVar("A")
B = TypeVar("B")

@dataclass(frozen=True)
class Point:
    x: Numeric
    y: Numeric

    @property
    def index(self) -> int:
        if isinstance(self.x, int):
            return self.x
        else:
            return round(self.x)

    @property
    def values(self) -> Tuple[Numeric, Numeric]:
        return (self.x, self.y)


def findContiguousRegions(oneDimImage: np.ndarray) -> Iterable[Tuple[int, int]]:
    """
    ex: |---###--#-----#####|  (where # is on, - is off)
        |0123456789...      |
    returns [(3,5), (8,8), (14,18)]
    """
    locations = []
    start = None
    for index, pixel in enumerate(oneDimImage):
        if pixel > 0 and start is None:
            start = index
        elif pixel == 0 and start is not None:
            locations.append((start, index))
            start = None

    return locations


def findContiguousRegionCenters(oneDimImage: np.ndarray) -> Iterable[int]:
    """
    ex: |---###--#-----#####|  (where # is on, - is off)
        |0123456789...      |
    returns [4, 8, 16]

    Rounds down to the nearest int
    """
    return [int(np.mean(list(locationPair))) for locationPair in findContiguousRegions(oneDimImage)]


def euclideanDistance(x: Numeric, y: Numeric) -> float:
    return sqrt((x**2) + (y**2))


def distanceBetweenPoints(firstPoint: Point, secondPoint: Point) -> float:
    return euclideanDistance(firstPoint.x - secondPoint.x, firstPoint.y - secondPoint.y)


def angleFromOffsets(x: Numeric, y: Numeric) -> float:
    return asin(y/euclideanDistance(x,y)) / pi * 180


def angleBetweenPoints(firstPoint: Point, secondPoint: Point) -> float:
    deltaX = secondPoint.x - firstPoint.x
    deltaY = secondPoint.y - firstPoint.y
    return angleFromOffsets(deltaX, deltaY)


def angleSimilarity(firstAngle: float, secondAngle: float) -> float:
    return (180 - abs(secondAngle - firstAngle)) / 180


def searchArea(initialRow: int, radius: int) -> Iterable[Tuple[int, int]]:
    area = []
    for column in range(1, radius+1):
        verticalOffset = 0
        while euclideanDistance(column, verticalOffset + 1) <= float(radius):
            verticalOffset += 1
        area.append((initialRow - verticalOffset, initialRow + verticalOffset))

    return area


def getPointLocations(image: np.ndarray) -> List[List[Point]]:
    columns = np.swapaxes(image, 0, 1)

    pointLocations = []

    # Scan horizontally across the image
    for column, pixels in enumerate(columns):
        # Get all points that could be part of a signal
        rows = findContiguousRegionCenters(pixels)
        points = common.mapList(rows, lambda row: Point(column, row))

        pointLocations.append(points)

    return pointLocations


# TODO: Make score multiply, or normalize the score by the length of the path
def score(currentPoint: Point, candidatePoint: Point, candidateAngle: float) -> float:
    DISTANCE_WEIGHT = .5

    currentAngle = angleBetweenPoints(candidatePoint, currentPoint)
    angleValue = 1 - angleSimilarity(currentAngle, candidateAngle)
    distanceValue = distanceBetweenPoints(currentPoint, candidatePoint)

    # print(currentAngle, candidateAngle, angleValue)
    # print(distanceValue)

    return (distanceValue * DISTANCE_WEIGHT) + (angleValue * (1 - DISTANCE_WEIGHT))


def getAdjacent(pointsByColumn, bestPathToPoint, startingColumn: int, minimumLookBack: int):
    rightColumnIndex = startingColumn
    leftColumnIndex = int(common.lowerClamp(startingColumn-minimumLookBack, 0))

    result = list(common.flatten(pointsByColumn[leftColumnIndex:rightColumnIndex]))

    while len(result) == 0 and leftColumnIndex >= 0:
        leftColumnIndex -= 1
        result = list(common.flatten(pointsByColumn[leftColumnIndex:rightColumnIndex])) # TODO: Make this more efficient?

    for point in result:
        assert point in bestPathToPoint, "Found point that hasn't yet been frozen"
        pointScore, _, pointAngle = bestPathToPoint[point]
        yield pointScore, point, pointAngle


def interpolate(fromPoint: Point, toPoint: Point) -> Iterator[Point]:
    slope = (toPoint.y - fromPoint.y) / (toPoint.x - fromPoint.x)
    f = lambda x: slope * (x - toPoint.x) + toPoint.y

    for x in range(fromPoint.index + 1, toPoint.index):
        yield Point(x, f(x))


def convertPointsToSignal(points: List[Point], width: Optional[int] = None) -> np.ndarray:
    assert len(points) > 0

    firstPoint = points[0]  # farthest from y-axis (recall we `back`-tracked earlier so paths are reversed)
    lastPoint = points[-1]  # closest to y-axis

    arraySize = width or (firstPoint.x + 1)
    signal = np.full(arraySize, np.nan, dtype=float)

    signal[firstPoint.index] = firstPoint.y
    priorPoint = firstPoint

    for point in points[1:]:
        if isnan(signal[point.index + 1]):
            for interpolatedPoint in interpolate(point, priorPoint):
                signal[interpolatedPoint.index] = interpolatedPoint.y

        signal[point.index] = point.y
        priorPoint = point

    return signal


def extractSignal(binary: BinaryImage) -> Optional[np.ndarray]:
    pointsByColumn = getPointLocations(binary.data)
    points = list(common.flatten(pointsByColumn))

    if len(points) == 0:
        return None

    minimumLookBack = 1

    bestPathToPoint: Dict[Point, Tuple[float, Optional[Point], float]] = {}

    # TODO: Allow some leeway either (1) Initialize the first N columns with 0s or (2) Search until some threshold for seeding is met

    # Initialize the DP table with base cases (far left side)
    # TODO: Is this even needed? See below. Currently causes error with `getAdjacent` if removed.
    for column in pointsByColumn[:1]:
        for point in column:
            bestPathToPoint[point] = (0, None, 0)

    # Build the table
    for column in pointsByColumn[1:]:
        for point in column:
            # Gather all other points in the perview of search for the current point
            adjacent = list(getAdjacent(pointsByColumn, bestPathToPoint, point.index, minimumLookBack))

            if len(adjacent) == 0:
                print(f"None adjacent to {point}")
                bestPathToPoint[point] = (0, None, 0)
            else:
                bestScore: float
                bestPoint: Point
                bestScore, bestPoint = min(
                    [(score(point, candidatePoint, candidateAngle) + cadidateScore, candidatePoint)
                    for cadidateScore, candidatePoint, candidateAngle in adjacent],
                    key=lambda triplet: triplet[0] # Just minimize by the score
                )
                bestPathToPoint[point] = (bestScore, bestPoint, angleBetweenPoints(bestPoint, point))

    # print(bestPathToPoint)

    # TODO: Search backward in some 2D area for the best path ?
    OPTIMAL_ENDING_WIDTH = 20
    optimalCandidates = list(getAdjacent(pointsByColumn, bestPathToPoint, startingColumn=binary.width, minimumLookBack=OPTIMAL_ENDING_WIDTH))

    # if len(optimalCandidates) == 0:

    _, current = min(
        [(totalScore, point) for totalScore, point, _ in optimalCandidates],
        key=lambda pair: pair[0] # Just minimize by the score
    )

    bestPath = []

    while current is not None:
        bestPath.append(current)
        _, current, _ = bestPathToPoint[current]

    signal = convertPointsToSignal(bestPath) #, width=binary.width)

    # scores = [bestPathToPoint[point][0] ** .5 for point in points]
    # plt.imshow(binary.toColor().data, cmap='Greys')
    # plt.scatter([point.x for point in points], [point.y for point in points], c=scores)
    # # plt.plot(signal, c='purple')
    # plt.show()

    return signal

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

    return ColorImage(output)

#######################################
# 
########################################

def process_digitized_signal(digitized_signal, real_signal,  
                             baseline=0.005, y_scale=0.0064, x_scale=2.54
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

from signal_detection import adaptive

def get_final_signal(*, label_path: str, output_path: str, save_image: bool, image_path=None, image=None):
    '''get path of a single image and return the digitized signal and its correlation with the real signal'''
    if image_path:
        # check if we have .png or .bmp image_path
        if image_path.split('.')[-1] == 'png':
            image_path = Path(image_path)
            image = openImage(image_path, png=True)
            binary = adaptive(image)

        elif image_path.split('.')[-1] == 'bmp':
            image_path = Path(image_path)
            binary =  openImage(image_path, bmp=True)

            

    # image = openImage(image_path)       # this is a binary image
    binary = signal_detection.adaptive(image)
    # image = binary
    signal = extractSignal(binary)
    # print(f'frst signal shape {signal.shape}')
    signal = ecgdigitize.digitizeSignal(image)
    # print(f'second signal shape {signal.shape}')
    with open(label_path, 'r') as f:
        real_signal = json.load(f)

    # try:
    if save_image:
        output = overlaySignalOnImage(signal, image.toColor())
        output_image_path = os.path.join(output_path, Path(image_path).name)
        saveImage(output, Path(output_image_path))
    
    # x an y values of the digitized signal after processing
    # x_vals, y_vals = process_digitized_signal(signal, real_signal)
    # # print(signal.shape, real_signal.__len__())
    # print(len(y_vals), len(real_signal))
    # plt.plot(y_vals, 'r')
    plt.plot(real_signal, 'b')
    plt.show()
    correlation = pearsonr(real_signal, signal)[0]
    print(f'corr: {correlation}')

    return correlation
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
                     image_path=r'datasets\image_dataset_v7.0\digitization\train\images\0_3by1_lead_0.png',
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