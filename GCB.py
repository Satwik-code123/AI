import cv2
import numpy as py

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("There was an error. Try again")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        print("Image cannot be captured, Please try again")
        break
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    lower_skin = py.array([0,20,70],dtype=py.uint8)
    upper_skin = py.array([20,255,255],dtype=py.uint8)

    mask = cv2.inRange(hsv,lower_skin,upper_skin)
    result = cv2.bitwise_and(frame,frame,mask=mask)

    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_cont = max(contours,key= cv2.contourArea)
        if contours:
            x,y,w,h = cv2.boundingRect(max_cont)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            centerx = int(x+w/2)
            centery = int(y+h/2)
            cv2.circle(frame,(centerx,centery),5,(0,255,0),2)
    
    cv2.imshow("Original Frame",frame)
    cv2.imshow("Filtered Image",result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break