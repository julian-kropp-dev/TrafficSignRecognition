#Steuerung des PiCars
Die Steuerung des PiCars geschieht über den verbauten RaspberryPi 4. 
Die Erkennung und Zuordnung der Verkehrsschilder geschieht aus Leistungsgründen **nicht** auf dem Raspberry, sondern auf einem externen Rechner.

Die folgenden Dateien bauen eine Webseite auf, welche mit der Hilfe von *request-Befehlen* das Auto fernsteuert:
- *backwheels.py*: Steuerung der Hinterräder
- *index.html*: Die Webseite über die man das Auto fahren lassen kann
- *robo.php*: Das PHP-Skript, welches die request-Befehle weiterverarbeitet
- *steering.py*: Steuerung des vorderen Servomotor für die Vorderräder

Diese Dateien sind außschließlich auf dem Pi zu installieren und lassen sich bei mir unter dem Dateipfad /var/www/html auf diesem finden.
Die restlichen Dateien zur Zuordnung der Verkehrsschilder liegen auf einem externen Laptop/PC und werden über den Dateipfad TrafficSignRecognition/src/TSR/src/ eingebunden.