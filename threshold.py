# checks if the lower and upper bounds work
#kashish_agarwal
import cv2
import numpy as np
import time
w = 100
h = 200
x1 = 117
y = 103
lower_bound = np.array([0,80,70])
upper_bound = np.array([40,140,130])
values = np.array([])

cap = cv2.VideoCapture(0)
def centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/(M['m00']+1))
    cy = int(M['m01']/(M['m00']+1))
    return (cx, cy)

drawpts=[]


while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #frame = cv2.GaussianBlur(frame, (5,5),0)
    kernel = np.ones((7, 7), np.uint8)
    frame = cv2.erode(frame,kernel,iterations = 2)
    #frame = cv2.dilate(frame, kernel, iterations=2)
    #frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_contours=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>800:
            res=cv2.drawContours(res,[cnt],-1,(0,225,0),3)
            best_contours.append(cnt)


    if len(best_contours)>0:
        cent = []
        for cnt in best_contours:
            cent.append(centroid(cnt))
        mini=min(cent,key = lambda x: x[1])
        print(mini)
        drawpts.append(mini)
    else:
        print("no contours")

    for pt in drawpts:
        cv2.circle(frame,pt,3,(0,255,0),1)

    cv2.imshow('res', res)
    cv2.imshow('drawing',frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()