import cv2
import time
import datetime
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
ledPin = 22
GPIO.setup(ledPin, GPIO.OUT)
GPIO.output(ledPin, GPIO.LOW) 

cam1 = cv2.VideoCapture(0)
#cam2 = cv2.VideoCapture(2)
frame_width1 = int(cam1.get(3))
frame_height1 = int(cam1.get(4))
#frame_width2 = int(cam2.get(3))
#frame_height2 = int(cam2.get(4))

frame_size1 = (frame_width1, frame_height1)
#frame_size2 = (frame_width2, frame_height2)
writer1 = cv2.VideoWriter('/home/pi/Desktop/Senior_Design_Project/test1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, frame_size1)
#writer2 = cv2.VideoWriter('/home/pi/Desktop/Senior_Design_Project/test2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, frame_size2)
count = 0
while True:
    ret1, image1 = cam1.read()
    #ret2, image2 = cam2.read()
    #hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
    #out.write(hsv)
    writer1.write(image1)
    #writer2.write(image2)
    count += 1
    if count == 1:
        print("camera start")
    GPIO.output(ledPin, GPIO.HIGH)
    #time.sleep(0.5)
    GPIO.output(ledPin, GPIO.LOW) 
    #time.sleep(0.5)
    #cv2.imshow('Web1', image1)
    #cv2.imshow('Web2', image2)
    
cam1.release()
#cam2.release()
writer1.release()
#writer2.release()
cv2.destroyAllWindows()
GPIO.output(ledPin, GPIO.LOW) 
time.sleep(0.5)
print("camera stop")