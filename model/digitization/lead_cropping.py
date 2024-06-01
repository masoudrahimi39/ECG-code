import cv2
import os
from dotenv import load_dotenv
import os
import json

def extract_subimages(image_path, text_path, output_dir, classes):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_height, img_width = image.shape[:2]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the text file
    with open(text_path, 'r') as file:
        lines = file.readlines()

    # Process each line in the text file
    for i, line in enumerate(lines):
        parts = line.strip().split()
        
        if len(parts) != 5:
            print(f"Skipping invalid line {i + 1}: {line}")
            continue

        try:
            cls, x_center, y_center, width, height = map(float, parts)

            # Only process lines where the class value is 0
            if cls != 0:
                continue

            # Calculate the coordinates of the bounding box
            x_center = int(x_center * img_width)
            y_center = int(y_center * img_height)
            width = int(width * img_width)
            height = int(height * img_height)

            # Increase width and height by 100%
            width = int(width * 2)
            height = int(height * 2)

            x_min = max(0, x_center - width // 2)
            y_min = max(0, y_center - height // 2)
            x_max = min(img_width, x_center + width // 2)
            y_max = min(img_height, y_center + height // 2)

            # Extract the sub-image
            sub_image = image[y_min:y_max, x_min:x_max]

            # Generate the output filename
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            lead_number = lines[i+1].strip().split()[0]
            lead_name = classes[int(lead_number)]
            output_path = os.path.join(output_dir, f'{name}_{lead_name}.png')

            # Save the sub-image
            cv2.imwrite(output_path, sub_image)
            print(f'Saved sub-image {i + 1} to {output_path}')

        except ValueError:
            print(f"Skipping invalid line {i + 1}: {line}")



if __name__ == "__main__":
    image_path = "datasets/image_dataset_v4.0/test/200_3by1.jpg" 
    text_path = "datasets/image_dataset_v4.0/test/200_3by1.txt" 
    output_dir = "datasets/single_lead_dataset/"  # Replace with your desired output directory

    load_dotenv()
    dataset_path = os.getenv("image_dataset_path")
    classes = json.loads(os.getenv("labels_dict"))
    classes = {v:k for k, v in classes.items()}

    extract_subimages(image_path, text_path, output_dir, classes)
