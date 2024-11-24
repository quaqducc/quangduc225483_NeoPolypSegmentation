import argparse
import cv2
import torch
import numpy as np
from segmentation_models_pytorch import UnetPlusPlus
import os

# Function to map masks to RGB (customize as per your color scheme)
def mask_to_rgb(mask, color_dict):
    h, w = mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in color_dict.items():
        rgb_image[mask == k] = color
    return rgb_image

def main(args):
    # Model setup
    model = UnetPlusPlus(
        encoder_name="resnet34",  # Encoder architecture
        encoder_weights=None,    # No pre-trained weights
        in_channels=3,           # RGB input
        classes=3                # Number of classes (adjust if necessary)
    )

    # Load checkpoint
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load image
    input_path = args.image_path
    ori_img = cv2.imread(input_path)
    if ori_img is None:
        print(f"Error: Unable to load image at {input_path}")
        return

    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]

    # Preprocess image
    trainsize = 256  # Resize to match the model's input size
    img = cv2.resize(ori_img, (trainsize, trainsize))
    img = img / 255.0  # Normalize
    img = np.transpose(img, (2, 0, 1))  # Channels first
    img_tensor = torch.tensor(img).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor).squeeze(0).cpu().numpy()
    output = np.argmax(output, axis=0)

    # Resize mask to original dimensions
    mask_resized = cv2.resize(output, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

    # Convert mask to RGB
    color_dict = {
        0: [0, 0, 0],       # Background
        1: [255, 0, 0],     # Class 1
        2: [0, 255, 0],     # Class 2
    }
    mask_rgb = mask_to_rgb(mask_resized, color_dict)

    # Save the segmented output
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "segmented_output.png")
    mask_rgb_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
    cv2.imwrite(output_path, mask_rgb_bgr)
    print(f"Segmented image saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for segmentation.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    main(args)
