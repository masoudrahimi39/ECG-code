from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()
dataset_path = os.getenv("image_dataset_path")

def make_config():
    # making the configuration file
    import yaml

    configuration={
    "path": dataset_path,
    "train": dataset_path + "/train",
    "test": dataset_path + "/test",
    "val": dataset_path + "/val",
    "names": {
        0: "lead"
        1: "lead_name"
    }
    }

    config_path = os.path.join(dataset_path, "config.yaml")

    with open(config_path, "w") as file:
        documents = yaml.dump(configuration, file)
model = YOLO('yolov8n.pt')

results = model.train(data='coco8.yaml', epochs=100, imgsz=640)