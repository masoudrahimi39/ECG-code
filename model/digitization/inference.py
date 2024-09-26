from model.digitization.digitize import digitize_binary_image

if __name__=="__main__":
    from prepare_image_dataset.utils import crop_bb
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    import os
    import json
    import numpy as np
    import cv2

    load_dotenv()
    log_path = os.path.join(os.getenv("detection_dataset_path"), "train", "logs.json")

    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    sample_index = 0
    lead_index = 2

    sample = logs['samples'][sample_index]
    cropped_images = crop_bb(sample=sample, export_path=False, img_path=sample['save_path'], prefix="")

    cropped_image = np.array(cropped_images[lead_index])

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    binary_image = np.zeros_like(gray_image)
    binary_image[gray_image<10] = 255

    x_signal, y_signal = digitize_binary_image(binary_image, 130, 3.96)
    real_signal = sample['leads'][lead_index]['ecg']

    plt.subplot(311)
    plt.plot(x_signal, y_signal, label="digitized")
    plt.plot(real_signal, label="real")
    plt.legend()
    plt.subplot(312)
    plt.imshow(cropped_image)
    plt.savefig("tmp.png")
    plt.subplot(313)
    plt.imshow(binary_image)
    plt.savefig("tmp.png")
    plt.close("all")
