import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms

# Define the inference class
class SegmentationInference:
    def __init__(self, model_path, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=1):
        """Initialize the model and load the trained weights."""
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, activation=None
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(model_path, map_location=self.device)
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
        self.mean = torch.tensor(params["mean"]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(params["std"]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, image_path, img_size=(320, 320)):
        """Preprocess input image."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)
        image = image.astype(np.float32) / 255.0  # Normalize
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)  # HWC -> CHW, add batch dim
        image = (image - self.mean) / self.std  # Normalize using encoder params
        return image

    def predict(self, image_tensor):
        """Run inference on the image."""
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output)  # Convert logits to probabilities
            output = (output > 0.5).float()  # Apply thresholding
        return output.cpu().numpy().squeeze()

    def visualize(self, image_path):
        """Visualize the original image and predicted mask."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (320, 320))
        image_tensor = self.preprocess(image_path)
        mask = self.predict(image_tensor)
        
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
    image_path = "datasets/image_dataset_v7.0/segmentation/train/images/0_3by4_lead_1.png"
    
    segmenter = SegmentationInference(model_path)
    segmenter.visualize(image_path)
