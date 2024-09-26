from ultralytics import YOLO
import os
from dotenv import load_dotenv
from model.utils.utils import find_files_with_patterns
from PIL import Image
import matplotlib.pyplot as plt
import json

print("passed")


def inference(model, classes, img=None, img_path=None, crop=False, plot=False, scale_up_text_boxes=False):
    if type(model) == str:
        model = YOLO(model)
    if img_path:
        img = Image.open(img_path)

    res = model(img)
    out = []
    # Extract each bounding box and plot it
    for i, (x, y, w, h) in enumerate(res[0].boxes.xywh.cpu().numpy()):
        class_id = res[0].boxes[i].cls.cpu().numpy()[0]
        class_name = classes[class_id]

        
        if class_name != "lead" and scale_up_text_boxes:
            cropped_img = img.crop((x-w/2, y-h/2, x + w * 1.3, y + h/2))
        else:
            cropped_img = img.crop((x-w/2, y-h/2, x + w/2, y + h/2))


        out.append({
            "class_name": class_name,
            "bbox_xywh": (x, y, w, h),
            "img": cropped_img
        })

        if plot:
          title = f"predicted: {class_name}"

          plt.title(title)
          plt.imshow(cropped_img)
          plt.savefig(f'./model/detection/inf_img/{i}.jpg')
        #   plt.show()
          plt.close("all")

    return out




if __name__ == "__main__":
    load_dotenv()
    test_dataset_path = os.getenv("detection_dataset_path") +  '/test'
    best_model_path = os.getenv("detection_model")
    detection_classes = os.getenv("labels_dict")
    classes = json.loads(detection_classes)
    classes = {v:k for k, v in classes.items()}

    # get 2 files with the below pattern from the test dataset directoy
    patterns = ['_3by1.jpg', '_3by4.jpg', '_6by2.jpg', '_12by1.jpg']
    image_paths_for_inference = find_files_with_patterns(test_dataset_path, patterns, n_files=1)

    for img_path in image_paths_for_inference:
        out = inference(model=best_model_path, classes=classes, 
                        img_path=img_path, crop=True, plot=True, scale_up_text_boxes=True)
        print(type(out))
        print(len(out))
        print(out)
        break