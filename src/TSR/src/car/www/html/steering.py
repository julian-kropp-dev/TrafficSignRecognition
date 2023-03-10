import sys
import time
from adafruit_servokit import ServoKit


#übergebende Parameter in URL
type = sys.argv[1]

kit = ServoKit(channels=16)

#setzt Servos auf Nullpunk
kit.servo[0].angle = 180
kit.servo[0].angle = 0
kit.servo[0].angle = 0

#left
if type == "L":
    kit.servo[0].angle = 10
   

#right
if type == "R":
    kit.servo[0].angle = 80

#geradeaus ;)
if type == "G":
    kit.servo[0].angle = 45
