import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms
# from .data_loader import scale_up_img_size, resize_and_crop

def resize_and_crop(image, target_height=160, target_width=320, sliding_x=200):
    height, width = image.shape[:2]

    new_width = target_width
    new_height = target_height
    # Determine the scaling factor to resize the smaller dimension to 320
    # if width/target_width < height/target_height:
    #     new_width = target_width
    #     new_height = target_height
    #     # new_height = int((height / width) * target_width)
    # else:
    #     new_height = target_height
    #     new_width = int((width / height) * target_height)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # Determine the random crop coordinates
    x_max = new_width - target_width
    y_max = new_height - target_height

    y_start = 0
    if y_max > 0:
        raise Exception("I shouldn't be here!!!")

    if x_max > 0:
        xx = 0
        while xx<x_max:
            x_start = xx
            cropped_image = resized_image[y_start:y_start + target_height, x_start:x_start + target_width]
            yield cropped_image
            xx += sliding_x
    else:
        x_start = 0
        cropped_image = resized_image[y_start:y_start + target_height, x_start:x_start + target_width]
        yield cropped_image

# Define the inference class
class SegmentationInference:
    def __init__(self, model_path, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=1):
        """Initialize the model and load the trained weights."""
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, activation=None
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(model_path, map_location=self.device)['state_dict']

        model_dict = self.model.state_dict()
        missing_keys = []
        for key in model_dict.keys():
            key_in_checkpoint = 'model.'+key
            if key_in_checkpoint in checkpoint:
                model_dict[key] = checkpoint[key_in_checkpoint]
            else:
                missing_keys.append(key)
        
        self.model.load_state_dict(model_dict, strict=False)
        
        if missing_keys:
            print("Warning: The following keys were not found in the checkpoint:")
            for key in missing_keys:
                print(key)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get preprocessing parameters
        params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.mean = checkpoint['mean'].view(1, 3, 1, 1).to(self.device) # torch.tensor(params["mean"]).view(1, 3, 1, 1).to(self.device)
        # self.std = checkpoint['std'].view(1, 3, 1, 1).to(self.device) # torch.tensor(params["std"]).view(1, 3, 1, 1).to(self.device)
        self.mean = torch.tensor([[[[0.4850]],
                [[0.4560]],
                [[0.4060]]]]).to(self.device)
        self.std = torch.tensor([[[[0.2290]],
                [[0.2240]],
                [[0.2250]]]]).to(self.device)

        self.max_height = 320

    # def preprocess(self, image_path, img_size=(320, 320)):
    #     """Preprocess input image."""
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = cv2.resize(image, img_size)
    #     image = image.astype(np.float32) / 255.0  # Normalize
    #     image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)  # HWC -> CHW, add batch dim
    #     image = (image - self.mean) / self.std  # Normalize using encoder params
    #     return image

    def predict(self, image_tensor):
        """Run inference on the image."""
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output)  # Convert logits to probabilities
            output = (output > 0.5).float()  # Apply thresholding
        return output.cpu().numpy().squeeze()

    def preprocess(self, image_path, sliding_x, selected_size=(128, 320)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR-->RGB
        image = np.array(image)

        # image = cv2.resize(image, selected_size)

        # if image.shape[0] > self.max_height:
        #     # print(f"Skipping image {self.images_fps[i]} as its height {image.shape[0]} is greater than maximum height{self.max_height}")
        #     return self.__getitem__((i + 1) % len(self))

        # image = scale_up_img_size(image, divisible_by=32, target_height=self.max_height) # scale up image width and height

        # mask = Image.open(self.masks_fps[i]).convert('L')  # load single channel
        # mask = np.array(mask)
        # mask = cv2.resize(mask, selected_size)


        # mask = scale_up_img_size(mask, divisible_by=32, target_height=self.max_height)
        for image in resize_and_crop(image, target_height=selected_size[0], target_width=selected_size[1], sliding_x=sliding_x):
            image = image.transpose(2, 0, 1)
            image = image[np.newaxis, :]

            image = torch.tensor(image)

            # assert np.all(np.isin(mask, [0, 1])), "Mask contains non-binary values"

            # extract certain classes from mask (e.g. cars)
            # masks = [(mask == v) for v in self.class_values]
            # mask = np.stack(masks, axis=-1).astype("float")
            # mask = 1 - mask                                    # convert 1 to 0 and 0 to 1 since the background should be 0 and signal should be 1

            yield (image-self.mean)/self.std

    def __call__(self, image_path):
        masks = []
        
        image = cv2.imread(image_path)
        original_height, original_width = image.shape[:2]
        
        selected_size = (96, 960)
        sliding_x=selected_size[1]//2
        for image_tensor in self.preprocess(image_path, selected_size=selected_size, sliding_x=sliding_x):
            mask = self.predict(image_tensor)
            masks.append(mask)
        print("len of masks: ", len(masks))
        final_mask = np.zeros((selected_size[0], selected_size[1]*len(masks)-sliding_x*(len(masks)-1)))
        start_x = 0
        for mask in masks:
            final_mask[:, start_x: start_x+selected_size[1]] += mask.copy()
            start_x += sliding_x
        final_mask[final_mask>1]=1

        final_mask = cv2.resize(final_mask, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
        final_mask = np.clip(final_mask, 0, 1)

        return final_mask

    def visualize(self, image_path):
        """Visualize the original image and predicted mask."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (320, 320))

        mask = self(image_path)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_resized)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.savefig("img.png")
        plt.show()

# Example usage
if __name__ == "__main__":
    model_path = "model/segmentation/unet/unet-segmentation.pth"
    image_path = "model/segmentation/unet/img.png"
    
    segmenter = SegmentationInference(model_path)
    segmenter.visualize(image_path)
