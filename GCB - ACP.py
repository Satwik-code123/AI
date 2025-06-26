import cv2
import numpy as np

vcap = cv2.VideoCapture(0)
if not vcap.isOpened():
    print("There was an error in opening the camera, please try again")
    exit()

while True:
    v1,v2 = vcap.read()
    if not v1:
        print("Image has not been captured, please try again")
        break
    hsv = cv2.cvtColor(v2,cv2.COLOR_BGR2HSV)
    dark = np.array([0,20,70],dtype=np.uint8)
    light = np.array([20,255,255],dtype=np.uint8)

    mask = cv2.inRange(hsv,dark,light)
    result = cv2.bitwise_and(v2,v2,mask=mask)

    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_cont = max(contours,key= cv2.contourArea)
        if contours:
            x,y,w,h = cv2.boundingRect(max_cont)
            cv2.rectangle(v2,(x,y),(x+w,y+h),(255,255,0),2)
            cx = int(x+w/2)
            cy = int(y+h/2)
            cv2.circle(v2,(cx,cy),5,(255,255,0),2)

    cv2.imshow("Original Frame",v2)
    cv2.imshow("Filtered Image",result)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break