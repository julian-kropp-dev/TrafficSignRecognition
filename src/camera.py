import torch
import cv2

# Lade das trainierte Modell aus der .pth-Datei
model = torch.load("trained_model.pth")
model.eval()

# Initialisiere die Webcam
cap = cv2.VideoCapture(0)

while True:
    # Lese das aktuelle Frame von der Webcam
    ret, frame = cap.read()

    # Verarbeite das Frame (z.B. Größenänderung, normalisieren, etc.)
    processed_frame = frame

    # Führe das Frame durch das Modell
    output = model(processed_frame)

    # Interpretiere das Modell-Ergebnis und ordne es dem Verkehrszeichen zu
    sign = interpret_output(output)

    # Zeige das Ergebnis auf dem Bildschirm
    cv2.imshow("Webcam", sign)

    # Warte auf eine Tastatureingabe (z.B. 'q', um das Programm zu beenden)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Freigabe der Ressourcen
cap.release()
cv2.destroyAllWindows()