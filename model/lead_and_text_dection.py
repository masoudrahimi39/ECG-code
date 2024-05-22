from ultralytics import YOLO, settings
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

make_config()

# Update a setting; enable MLFlow
settings.update({'mlflow': True})
# os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////Users/rezakarbasi/mlflow/mlflow.db"

os.environ["MLFLOW_EXPERIMENT_NAME"] = "lead-training"

model = YOLO('./model/yolov8n.pt')
# print(next(os.walk("./")))
# ultralytics train doc https://docs.ultralytics.com/modes/train/#train-settings
# exist_ok = False
results = model.train(data='./model/config.yaml', epochs=1, imgsz=640, save=True, resume=False, batch=16,
                      project='./model/alaki_1', name='alaki_2', optimizer='auto', pose=0, kobj=0,
                      device='cpu',
                      exist_ok=False,
                      pretrained=False,
                       )