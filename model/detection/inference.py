from ultralytics import YOLO
import os
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt
import json

# todo: the below should be changed
import sys
utils_dir = os.path.normpath('d:/Education/4_ECG_digitization/ECG-code/model/utils')
sys.path.append(utils_dir)
from utils import find_files_with_patterns


def inference(model, classes, img=None, img_path=None, crop=False, plot_all=False, plot_croped=False, 
              save=False, save_dir=None, scale_up_text_boxes=False):
    ''' it returns a list containing a dictionary for each bounding box. if there are 3 bounding boxes in the
    image, so there will be a list with 3 elements. and each element is dictionary with these 
    keys: (class_name, bbox_xywh, PIL.Image.Image) '''

    if type(model) == str:
        model = YOLO(model)
    if img_path:
        img = Image.open(img_path)

    res = model(img)

    if plot_all:
        plt.imshow(res[0].plot())
        plt.show()
        plt.close("all")
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.imshow(res[0].plot())
        plt.savefig(f'{save_dir}/all.jpg')
        plt.close("all")
    
    
    if crop:
        out = []
        # Extract each bounding box and plot it
        for i, (x, y, w, h) in enumerate(res[0].boxes.xywh.cpu().numpy()):
            class_id = res[0].boxes[i].cls.cpu().numpy()[0]
            class_name = classes[class_id]

            # todo: when text bounding boxes fixed in dataset, this scale up should be removed
            if class_name != "lead" and scale_up_text_boxes:
                cropped_img = img.crop((x-w/2, y-h/2, x + w * 1.3, y + h/2))
            else:
                cropped_img = img.crop((x-w/2, y-h/2, x + w/2, y + h/2))

            out.append({
                "class_name": class_name,
                "bbox_xywh": (x, y, w, h),
                "img": cropped_img
            })

            if plot_croped:
                title = f"predicted: {class_name}"
                plt.title(title)
                plt.imshow(cropped_img)
                plt.show()
                plt.close("all")
            if save:
                title = f"predicted: {class_name}"
                plt.title(title)
                plt.imshow(cropped_img)
                plt.savefig(f'{save_dir}/{i}.jpg')
                plt.close("all")
        return out




if __name__ == "__main__":
    load_dotenv()
    test_dataset_path = os.getenv("detection_dataset_path") +  '/test'
    detection_model_path = os.getenv("detection_model")
    detection_classes = os.getenv("labels_dict")
    classes = json.loads(detection_classes)
    classes = {v:k for k, v in classes.items()}

    # get 2 files with the below pattern from the test dataset directoy
    patterns = ['_3by1.jpg', '_3by4.jpg', '_6by2.jpg', '_12by1.jpg']
    image_paths_for_inference = find_files_with_patterns(test_dataset_path, patterns, n_files=1)

    for img_path in image_paths_for_inference:
        out = inference(detection_model_path, classes, img_path=img_path, crop=True, plot_all=False, 
                        plot_croped=False, save=True, save_dir='./model/detection/inf_img', 
                        scale_up_text_boxes=False)


        for i in out:
            print(i)
            print('------------')
        break