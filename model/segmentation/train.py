from ultralytics import YOLO, settings
import os
import json
from dotenv import load_dotenv
import yaml


def make_config(dataset_path, classes):

    configuration={
        "path": os.path.abspath(dataset_path),
        "train": os.path.join(os.path.abspath(dataset_path), "train"),
        "test": os.path.join(os.path.abspath(dataset_path), "test"),
        "val": os.path.join(os.path.abspath(dataset_path), "val"),
        "names": classes,
    }

    with open(config_path, "w") as file:
        documents = yaml.dump(configuration, file)

def train_segmentation_model(dataset_path, config_path):

    classes = {"signal":0}
    classes = {v:k for k, v in classes.items()}

    make_config(dataset_path, classes)

    model = YOLO("yolov8n-seg.pt")
    results = model.train(data=config_path,
                        epochs=150,
                        save=True,
                        augment=False, # Enable augmentations
                        hsv_h=0.3,  # HSV-Hue augmentation (fraction)
                        hsv_s=0.7,    # HSV-Saturation augmentation (fraction)
                        hsv_v=0.5,    # HSV-Value augmentation (fraction)
                        degrees=45,  # Image rotation (+/- deg)
                        translate=0.1,
                        scale=0.5,    # Image scale (+/- gain)
                        shear=20,    # Image shear (+/- deg)
                        perspective=0.0005,
                        flipud=0.2,
                        fliplr=0.2,
                        bgr=0.2,
                        mosaic=0,
                        mixup=0,
                        copy_paste=0,
                        erasing=0,
                        crop_fraction=0,
                        # project='/content/drive/MyDrive/slight-version-ECG/model/digitization/',
                        name='segmentation',
                        # optimizer='auto', pose=0, kobj=0,
                        device='cuda',
                        # resume=False,
                        save_period=10,
                        # exist_ok=True,
                        # pretrained=True,
                        )

if __name__ == "__main__":
    load_dotenv()
    dataset_path = os.getenv("refinement_dataset_path")

    config_path = os.getenv("segmentation_train_config")

    # Update a setting; enable MLFlow
    settings.update({'mlflow': True})
    # os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////Users/rezakarbasi/mlflow/mlflow.db"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "segmentation-training"


    train_segmentation_model(dataset_path, config_path)