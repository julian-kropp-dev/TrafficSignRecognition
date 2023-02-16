import RPi.GPIO as GPIO
import time

# Pin-Definitionen
MOTOR_A = 17 # IN1 und IN2
MOTOR_B = 27# IN3 und IN4
STEERING = 0

# Pin-Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_A, GPIO.OUT)
GPIO.setup(MOTOR_B, GPIO.OUT)
GPIO.setup(STEERING, GPIO.OUT)

# Funktionen zum Vorwärtsfahren, Rückwärtsfahren und Anhalten
def forward():
    GPIO.output(MOTOR_A, GPIO.HIGH)
    GPIO.output(MOTOR_B, GPIO.HIGH)

def reverse():
    GPIO.output(MOTOR_A, GPIO.LOW)
    GPIO.output(MOTOR_B, GPIO.LOW)

def stop():
    GPIO.output(MOTOR_A, GPIO.LOW)
    GPIO.output(MOTOR_B, GPIO.LOW)

def steer(angle):
    duty_cycle = (angle / 180.0) * 10.0 + 2.5
    pwm = GPIO.PWM(STEERING, 50)
    pwm.start(duty_cycle)
    time.sleep(0.5)
    pwm.stop()

# Testfahrt
try:
    while True:
        # Vorwärtsfahren für 2 Sekunden
        forward()
        steer(90)
        time.sleep(2)

        # Anhalten für 1 Sekunde
        stop()
        time.sleep(1)

        # Rückwärtsfahren für 2 Sekunden
        reverse()
        steer(45)
        time.sleep(2)

        # Anhalten für 1 Sekunde
        stop()
        time.sleep(1)

# Programm beenden mit Strg + C
except KeyboardInterrupt:
    GPIO.cleanup()
