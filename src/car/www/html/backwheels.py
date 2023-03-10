import sys
import Adafruit_PCA9685
import RPi.GPIO as GPIO

speed = sys.argv[1]
type = sys.argv[2]

#PCA9685 ist das MainBoard in der Mitte
pwm = Adafruit_PCA9685.PCA9685()

#Gpio Pi auf Standard
GPIO.cleanup()

#Setzt Nummerierung der Pins auf BCM
GPIO.setmode(GPIO.BCM)

GPIO.setup((17, 27), GPIO.OUT)


#forward
if type == "F":
    GPIO.output(17, False)
    GPIO.output(27, False)

    pwm.set_pwm(4, 0, int(speed))
    pwm.set_pwm(5, 0, int(speed))

#backward
if type == "B":
    GPIO.output(17, True)
    GPIO.output(27, True)
    pwm.set_pwm(4, 0, int(speed))
    pwm.set_pwm(5, 0, int(speed))
