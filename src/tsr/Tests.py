import cv2
import pytest as pytest
import torch
import numpy as np
from Cam import build_model, class_names, transform

# Teste das Modell, indem es auf ein zufälliges Bild angewendet wird
def test_model():
    device = 'cpu'
    model_path = 'model.pth'

    # Ein zufälliges Bild erstellen
    image = np.random.randint(0, 255, size=(224, 224, 3)).astype('uint8')

    # Model laden
    model = build_model(num_classes=43).to(device)
    model = model.eval()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

    # Bildtransformationen anwenden
    image_tensor = transform(image=image)['image']
    image_tensor = image_tensor.unsqueeze(0)

    # Forward pass durch das Modell
    outputs = model(image_tensor.to(device))

    # Überprüfen, ob das Modell 43 Ausgaben hat (eine für jede Verkehrsschild-Klasse)
    assert outputs.shape[1] == 43

    # Überprüfen, ob die Summe aller Ausgaben ungefähr 1 ergibt
    assert torch.sum(torch.nn.functional.softmax(outputs, dim=1)).item() == pytest.approx(1, rel=1e-2)

    # Überprüfen, ob die Vorhersage des Modells eine der Verkehrsschild-Klassen ist
    class_idx = torch.argmax(outputs, dim=1)
    assert class_names[int(class_idx)] in class_names

# Teste, ob das Kamerabild eingelesen werden kann
def test_camera():
    cameraView = cv2.VideoCapture(0)
    _, image = cameraView.read()
    assert image is not None

# Teste, ob ein Logfile erstellt werden kann
def test_logfile():
    import os
    os.remove("log.txt")
    images_counter = 1
    class_idx = 5
    percent_value = 0.95
    with open("log.txt", "a") as f:
        f.write(f"####\n# Bild-Nr: {images_counter}, Erkanntes Schild: {str(class_names[int(class_idx)])}, Zu: {percent_value}%\n####")
    assert os.path.exists("log.txt")

if __name__ == "__main__":
    test_model()
    test_camera()
    test_logfile()