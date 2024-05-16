from ultralytics import YOLO
from dotenv import load_dotenv
import os
import json

load_dotenv()
dataset_path = os.getenv("image_dataset_path")
classes = json.loads(os.getenv("labels_dict"))
classes = {v:k for k, v in classes.items()}

def make_config():
    # making the configuration file
    import yaml

    configuration={
        "path": os.path.abspath(dataset_path),
        "train": os.path.abspath(dataset_path + "/train"),
        "test": os.path.abspath(dataset_path + "/test"),
        "val": os.path.abspath(dataset_path + "/val"),
        "names": classes
    }

    with open("./model/config.yaml", "w") as file:
        documents = yaml.dump(configuration, file)

model = YOLO('./yolov8n.pt')
print(next(os.walk("./")))
results = model.train(data='./model/config.yaml', epochs=100, imgsz=640)