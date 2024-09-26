from ultralytics import YOLO, settings
from dotenv import load_dotenv
import os
import json
import yaml

def make_config(dir_to_save_config, classes):
    configuration={
        "path": os.path.abspath(dataset_path),
        "train": os.path.abspath(dataset_path + "/train"),
        "test": os.path.abspath(dataset_path + "/test"),
        "val": os.path.abspath(dataset_path + "/val"),
        "names": classes, }
    with open(dir_to_save_config, "w") as file:
        documents = yaml.dump(configuration, file)


load_dotenv()
dataset_path = os.getenv("detection_dataset_path")
detection_classes = os.getenv("labels_dict")
config_train_dir = os.getenv("detection_confing_train")
base_model_detection_yolov8 = os.getenv("base_model_detection_yolov8")

classes = json.loads(detection_classes)
classes = {v:k for k, v in classes.items()}

make_config(config_train_dir, classes)

settings.update({'mlflow': True})
# os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////Users/rezakarbasi/mlflow/mlflow.db"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "lead-training"


model = YOLO(base_model_detection_yolov8)
results = model.train(data=config_train_dir, 
                        epochs=1,
                        device='cpu',
                        save=True,
                        augment=True, # Enable augmentations,
                        degrees=0,
                        mosaic=0, mixup=0, erasing=0, crop_fraction=0,
                        copy_paste=0, translate=0, perspective=0,
                        hsv_h=0.015,  # HSV-Hue augmentation (fraction)
                        hsv_s=0.7,    # HSV-Saturation augmentation (fraction)
                        hsv_v=0.4,    # HSV-Value augmentation (fraction)
                        scale=0.5,    # Image scale (+/- gain)
                        shear=0.0,    # Image shear (+/- deg)
                        bgr=0.1,
                        project='./model/detection/',
                        name='result',
                        optimizer='auto', pose=0, kobj=0,
                        save_period=25,
                        )