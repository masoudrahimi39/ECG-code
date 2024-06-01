from ultralytics import YOLO
import os
from dotenv import load_dotenv
import fnmatch


def find_files_with_patterns(directory, patterns, n_files=2):
    ''' return a list containg directories of `n_files` images from each lead format from the test dataset.'''

    matched_files = {pattern: [] for pattern in patterns}

    for file in os.listdir(directory):
        # Check if we've already found two files for this pattern
        for pattern in patterns:
            if len(matched_files[pattern]) >= n_files:
                continue  # Move to the next pattern if already found 2 files
            
            # Check if the file matches the current pattern
            if fnmatch.fnmatch(file, f"*{pattern}"):
                matched_files[pattern].append(test_dataset_path+'/'+file)
    
    image_paths_for_inference = []
    for pattern, files in matched_files.items():
        image_paths_for_inference.extend(files)

    return image_paths_for_inference


### load 2 image from each lead format for testing
load_dotenv()
test_dataset_path = os.getenv("image_dataset_path") +  '/test'
patterns = ['_3by1.jpg', '_3by4.jpg', '_6by2.jpg', '_12by1.jpg']
image_paths_for_inference = find_files_with_patterns(test_dataset_path, patterns, n_files=2)


### load model and make inference
model = YOLO('./model/alaki_1/alaki_22/weights/best.pt')
# results = model(source=image_paths_for_inference, show=False, show_labels=True, show_boxes=True, imgsz=320, conf=0.5, device='cpu')  # list of Results objects
results = model(image_paths_for_inference)  # list of Results objects

cnt = 0
for result in results:
    boxes = result.boxes                # Boxes object for bounding box outputs
    # masks = result.masks              # Masks object for segmentation masks outputs
    # keypoints = result.keypoints      # Keypoints object for pose outputs
    probs = result.probs                # Probs object for classification outputs
    obb = result.obb                    # Oriented boxes object for OBB outputs
    # result.show()                     # display to screen
    result.save(filename=f'result_{cnt}.jpg')  # save to disk
    cnt += 1
