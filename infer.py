import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import UnetPlusPlus
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
model = UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
).to(device)

# Load the model checkpoint
checkpoint_path = "unet_model.pth"  # Update with your model checkpoint path
checkpoint = torch.load(checkpoint_path, map_location=device)
if "module." in list(checkpoint["model"].keys())[0]:
    checkpoint["model"] = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}

# Load the state dict into the model
model.load_state_dict(checkpoint["model"])
model.eval()


# Transform for input image
val_transformation = Compose([
    Resize(256, 256),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Function to overlay the mask on the image
def overlay_mask(image, mask):
    color_dict = {
        0: (0, 0, 0),       # Background
        1: (255, 0, 0),     # Class 1 (Red)
        2: (0, 255, 0),     # Class 2 (Green)
    }

    # Create an RGB mask
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in color_dict.items():
        mask_rgb[mask == cls] = color

    # Blend the original image and the mask
    blended = cv2.addWeighted(image, 0.6, mask_rgb, 0.4, 0)
    return blended

# Function for inference on a single image
def infer_one_image(image_path):
    # Load and preprocess the image
    image_name = os.path.basename(image_path)
    ori_img = cv2.imread(image_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]

    # Resize and transform
    resized_img = cv2.resize(ori_img, (256, 256))
    transformed = val_transformation(image=resized_img)
    input_img = transformed["image"].unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_img).squeeze(0).cpu()
        output_mask = F.softmax(output, dim=0).numpy()
        mask = np.argmax(cv2.resize(output_mask.transpose(1, 2, 0), (ori_w, ori_h)), axis=-1)

    # Overlay the mask on the original image
    result = overlay_mask(ori_img, mask)

    # Ensure the prediction directory exists
    os.makedirs("prediction", exist_ok=True)

    # Save and display the result
    output_path = os.path.join("prediction", image_name)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)


# Main function for command-line usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    infer_one_image(args.image_path)
