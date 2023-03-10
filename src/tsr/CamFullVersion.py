import time

import cv2
import numpy as np
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from torch import topk
from model import build_model

device = 'cpu'
sign_names_df = pd.read_csv('signnames.csv')
class_names = sign_names_df.SignName.tolist()
model_path = 'model.pth'

images_counter = 0
cameraView = cv2.VideoCapture(0)

model = build_model(num_classes=43).to(device)
model = model.eval()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def apply_color_map(CAMs, width, height, orig_image):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        result = cv2.resize(result, (224, 224))
        return result

def visualize_and_save_map(result, orig_image, gt_idx=None, class_idx=None):
    if class_idx is not None:
        text = f"Schild: {str(class_names[int(class_idx)])}"
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(result, (5, 20 - int(text_height*1.5)), (5 + text_width, 30), (255, 255, 255), cv2.FILLED)
        cv2.putText(
            result,
            text, (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            cv2.LINE_AA
        )
    orig_image = cv2.resize(orig_image, (224, 224))
    img_concat = cv2.hconcat([
        np.array(result, dtype=np.uint8),
        np.array(orig_image, dtype=np.uint8)
    ])
    cv2.imshow('Live-Bild: Julians Verkehrsschilderkennung', img_concat)
    cv2.waitKey(1)

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model._modules.get('features').register_forward_hook(hook_feature)
# Get the softmax weight.
params = list(model.parameters())
weight_softmax = np.squeeze(params[-4].data.cpu().numpy())

# Define the transforms, resize => tensor => normalize.
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
    ])

#main-program
if __name__ == '__main__':
    while True:
        # Read the image.
        _, image = cameraView.read()
        orig_image = image.copy()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = orig_image.shape
        # Apply the image transforms.
        image_tensor = transform(image=image)['image']
        # Add batch dimension.
        image_tensor = image_tensor.unsqueeze(0)
        # Forward pass through model.
        outputs = model(image_tensor.to(device))
        # Get the softmax probabilities.
        probs = F.softmax(outputs).data.squeeze()
        per = torch.nn.functional.softmax(outputs, dim=1)
        percent, _ = per.topk(1, dim=1)
        percent_value = percent.item()
        # Get the class indices of top k probabilities.
        class_idx = topk(probs, 1)[1].int()
        # Generate class activation mapping for the top1 prediction.
        CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
        # Show and save the results.
        result = apply_color_map(CAMs, width, height, orig_image)
        visualize_and_save_map(result, orig_image, None, class_idx)
        images_counter += 1
        if percent_value > 0.95:
            print(f"Bild-Nr: {images_counter}, Erkannt: {str(class_names[int(class_idx)])}")

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()