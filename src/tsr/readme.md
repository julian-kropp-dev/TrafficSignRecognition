# Steuerung der Verkehrszeichenerkennung

Die Erkennung und Zuordnung der Verkehrsschilder erfolgt aus Leistungsgründen **nicht** auf dem RaspberryPi, sondern auf einem Laptop/PC.
Der RaspberryPi hat einfach nicht genug Grafik-Leistung, um eine halbwegs flüssige Erkennung zu gewährleisten.

Folgende Dateien befinden sich in diesem Ordner:
- *Cam.py*: Zuordnung der Schilder und senden von requests an den Raspberry zur Steuerung des PiCars
- *log.txt*: In diesem Log-File werden die erkannten Verkehrsschilder zur späteren Analyse des Systems gespeichert
- *model.pth*: Das vortrainierte Model
- *model.py*: Enthält eine Funktion "build_model", die das Model erstellt
- *signnames.csv*: Tabelle mit allen Verkehrsschilder, die erkannt werden können
- *Tests.py*: Die geschriebenen Unit- und Software-Integrations-Tests

Copyright:
Die model.pth und model.py Datei stammen von XXX
