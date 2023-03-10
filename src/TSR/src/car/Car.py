import Adafruit_PCA9685
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit

pwm = Adafruit_PCA9685.PCA9685()
frontwheel = ServoKit(channels=16)

GPIO.cleanup()

GPIO.setmode(GPIO.BCM)
GPIO.setup((17, 27), GPIO.OUT)

frontwheel.servo[0].angle = 0

def back_Forward(speed):
    GPIO.output(17, False)
    GPIO.output(27, False)

    pwm.set_pwm(4, 0, speed)
    pwm.set_pwm(5, 0, speed)

def back_Brake(speed):
    GPIO.output(17, True)
    GPIO.output(27, True)

    pwm.set_pwm(4, 0, speed)
    pwm.set_pwm(5, 0, speed)

def drive_Forward():
    frontwheel.servo[0].angel = 45

def drive_Right():
    frontwheel.servo[0].angel = 80

def drive_Left():
    frontwheel.servo[0].angel = 10
