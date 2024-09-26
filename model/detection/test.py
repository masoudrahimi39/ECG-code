from ultralytics import YOLO, settings
from dotenv import load_dotenv
import os
import json
import yaml

def make_config(dir_to_save_config, classes):
    configuration={
        "path": os.path.abspath(dataset_path),
        "test": os.path.abspath(dataset_path + "/test"),
        "train": os.path.abspath(dataset_path + "/train"),
        "val": os.path.abspath(dataset_path + "/val"),
        "names": classes,}
    with open(dir_to_save_config, "w") as file:
      documents = yaml.dump(configuration, file)


load_dotenv()
dataset_path = os.getenv("image_dataset_path")
detection_classes = os.getenv("labels_dict")
config_test_dir = os.getenv("detection_confing_test")
best_model_path = os.getenv("best_model_detection")


classes = json.loads(detection_classes)
classes = {v:k for k, v in classes.items()}

make_config(config_test_dir, classes)

settings.update({'mlflow': True})
# os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////Users/rezakarbasi/mlflow/mlflow.db"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "lead-testing"



model = YOLO(best_model_path)
test_results = model.val(
    data=config_test_dir,
    save_json=True,
    split='test',
    max_det = 26,
    plots=True,
)

print(f'mAP50-95: {test_results.box.map}')  # mAP50-95
print(f'mAP50: {test_results.box.map50}')  # mAP50
print(f'mAP75: {test_results.box.map75}')  # mAP75
print(test_results.box.maps)
