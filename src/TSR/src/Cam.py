from datetime import datetime

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
# VideoCapture(0): Interne Kamera verwenden
# VideoCapture(1): Externe Kamera verwenden
cameraView = cv2.VideoCapture(0)

# Model laden
model = build_model(num_classes=43).to(device)
model = model.eval()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

# CNN-Gewichte initialisieren
params = list(model.parameters())
weight_softmax = np.squeeze(params[-4].data.cpu().numpy())

# Bildtransformationen: Zuschneiden, normalisieren, konvertieren
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])

# Hier beginnt das Programm
if __name__ == '__main__':
    while True:
        # Live-Kamerabild einlesen
        _, image = cameraView.read()
        orig_image = image.copy()
        height, width, _ = orig_image.shape
        # Bildtransformationen anwenden
        image_tensor = transform(image=image)['image']
        image_tensor = image_tensor.unsqueeze(0)
        # Forward pass through model
        outputs = model(image_tensor.to(device))
        # Prozentuale Wahrscheinlichkeit des erkannten Verkehrsschildes
        probs = F.softmax(outputs).data.squeeze()
        per = torch.nn.functional.softmax(outputs, dim=1)
        percent, _ = per.topk(1, dim=1)
        percent_value = percent.item()
        # Berechnet Klasse mit höchster Wahrscheinlichkeit
        class_idx = topk(probs, 1)[1].int()
        # Bildcounter erhöhen
        images_counter += 1

        f = open("log.txt", "a")
        f.write(f"####\n# {datetime.now()} Bild-Nr: {images_counter}, Erkanntes Schild: {str(class_names[int(class_idx)])}, Zu: {percent_value}%\n####")
        f.close()

        # nur wenn die Wahrscheinlichkeit über 90% liegt, damit weiterarbeiten
        if percent_value > 0.90:
            trafficSign = str(class_names[int(class_idx)])
            if trafficSign == "120km/h":
                # back_F(1000)
                # drive_F()
                print(
                    f"####\n# Bild-Nr: {images_counter}, Erkanntes Schild: {str(class_names[int(class_idx)])}, Zu: {percent_value}%\n####")
            elif trafficSign == "30km/h":
                # back_F(700)
                # drive_F()
                print(
                    f"####\n# Bild-Nr: {images_counter}, Erkanntes Schild: {str(class_names[int(class_idx)])}, Zu: {percent_value}%\n####")
            elif trafficSign == "50km/h":
                # back_F(850)
                # drive_F()
                print(
                    f"####\n# Bild-Nr: {images_counter}, Erkanntes Schild: {str(class_names[int(class_idx)])}, Zu: {percent_value}%\n####")
            elif trafficSign == "Durchfahrt verboten":
                # back_F(0)
                # back_B(500)
                # time.sleep(1)
                # drive_L()
                # back_F(500)
                # time.sleep(5)
                print(
                    f"####\n# Bild-Nr: {images_counter}, Erkanntes Schild: {str(class_names[int(class_idx)])}, Zu: {percent_value}%\n####")
            elif trafficSign == "Vorgeschriebene Fahrtrichtung links":
                # drive_L()
                print(
                    f"####\n# Bild-Nr: {images_counter}, Erkanntes Schild: {str(class_names[int(class_idx)])}, Zu: {percent_value}%\n####")
            elif trafficSign == "Vorgeschriebene Fahrtrichtung rechts":
                # drive_R()
                print(
                    f"####\n# Bild-Nr: {images_counter}, Erkanntes Schild: {str(class_names[int(class_idx)])}, Zu: {percent_value}%\n####")
            elif trafficSign == "Stopp":
                # back_F(0)
                # time.sleep(3)
                # back_F(500)
                print(
                    f"####\n# Bild-Nr: {images_counter}, Erkanntes Schild: {str(class_names[int(class_idx)])}, Zu: {percent_value}%\n####")
