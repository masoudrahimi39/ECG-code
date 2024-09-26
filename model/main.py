from dotenv import load_dotenv
from ultralytics import YOLO
from model.utils.utils import find_files_with_patterns
import os

# todo: after completing this script, it should be converted to function(s).

decetion_model_path = os.getenv("best_model_detection")
segmentation_model_path = os.getenv("best_model_detection")
detection_model = YOLO(decetion_model_path)
segmentation_model = YOLO(segmentation_model_path)

detection_classes = os.getenv("labels_dict")
classes = json.loads(detection_classes)
classes = {v:k for k, v in classes.items()}

test_dataset_path = os.getenv("detection_dataset_path") +  '/test'
patterns = ['_3by1.jpg', '_3by4.jpg', '_6by2.jpg', '_12by1.jpg']
img_pathes = find_files_with_patterns(test_dataset_path, patterns, n_files=1)

for img_path in image_paths_for_inference:
    
    # out is a list containing a dictionary for each box in the image provided
    out = inference(detection_model, classes, img_path=img_path, crop=True, plot_all=False, 
                    plot_croped=False, save=True, save_dir='./model/inf_img', 
                    scale_up_text_boxes=False)

    # todo: in the main.ipynb there is the below code. where should it be added? 
        # parsed = make_detection_output(image_path, res, plot=True, bigger_text_box=True)
        # parsed_texts = [p for p in parsed if p['class_name'] != 'lead']
        # parsed_leads = [p for p in parsed if p['class_name'] == 'lead']
    # todo: perform easy ocr

    # todo: compare confidence of easy oce and yolo classification to predict the label name

    # todo: add lead_and_text matching

    # todo: perform segemtation to get the mask

    # todo: digitize the binary image and get signal

    # todo: compare digitized and the real signal

            
                        
