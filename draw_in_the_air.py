#MY REFERENCE:  https://learning.oreilly.com/videos/hands-on-opencv-4/9781789618464/9781789618464-video3_6
#FOR OTHER REFERENCES: https://docs.google.com/document/d/1HO3WUC97w5CL5iXMvXOGLU-5_NzgXTCxn08paHcIh60/edit
#Make sure colour of the hand does not coincide with the colour of the
#background.
#Place your hand inside the rectangle and press "a" to start drawing .
#Press "x" to stop drawing.
import cv2
import numpy as np
import math
#This function captures the histogram of hand and returns the histogram.
def capture_histogram(source):
    cap=cv2.VideoCapture(source)
    while cap.isOpened():
        _,frame=cap.read()
        frame=cv2.flip(frame,1)
        frame=cv2.resize(frame,(1000,600))
        cv2.putText(frame,"put hand inside the box and press 'a'",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(500,300),(600,400),(0,0,0),2)
        box=frame[300:400,500:600]
        cv2.imshow('to_capture_histogram',frame)
        k=cv2.waitKey(1)
        if k==ord('a'):
            object_color=box
            cap.release()
            cv2.destroyAllWindows()
            break
        if k==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    object_color_hsv=cv2.cvtColor(object_color,cv2.COLOR_BGR2HSV)
    object_hist=cv2.calcHist([object_color_hsv],[0,1],None,[12,15],[0,180,0,256])
    cv2.normalize(object_hist,object_hist,0,255,cv2.NORM_MINMAX)
    return object_hist
#This function takes two inputs the frame (which is the live video in this case) and the
#histogram that is to be located in the frame(histogram of hand in this case) given,and
#returns only the part of frame that is located in the histogram.
def locate_object(frame,object_hist):
    hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    object_segment=cv2.calcBackProject([hsv_frame],[0,1],object_hist,[0,180,0,256],1)
    _,segment_thresh=cv2.threshold(object_segment,100,255,cv2.THRESH_BINARY)
    kernel=None
    disc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    filtered = cv2.filter2D(segment_thresh, -1, disc)
    eroded = cv2.erode(filtered, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    masked = cv2.bitwise_and(frame, frame, mask=closing)
    return closing, masked, segment_thresh
#This function two inputs tha frame(which is live video in this case) and histogram of the
#hand detected.And this function draws the contour around the hand and returns the
#dictionary called "return_value".
def detect_hand(frame,hist):
    return_value={} #return_value is an empty dictionary.
    detect_hand,masked,raw=locate_object(frame,hist)
    return_value['binary']=detect_hand
    return_value['masked'] = masked
    return_value['raw'] = raw
    contours,_=cv2.findContours(detect_hand,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas=[cv2.contourArea(c) for c in contours]
    largest_contour_index=np.argmax(areas)
    palm_area=cv2.contourArea(contours[largest_contour_index])
    cnt=contours[largest_contour_index]
    if largest_contour_index is not None and palm_area>10000:
        cnt=contours[largest_contour_index]
        return_value['contours']=cnt
        cpy=frame.copy()
        cv2.drawContours(cpy,[cnt],0,(0,255,0),3)
        return_value['boundaries']=cpy
        return True,return_value
    else:
        return False,return_value
#This function returns the finger tips of the hand detected.
def extract_finger_tips(hand):
    cnt=hand['contours']
    points=[]
    hull=cv2.convexHull(cnt,returnPoints=False)
    defects=cv2.convexityDefects(cnt,hull)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        end = tuple(cnt[e][0])
        # print(end)
        points.append(end)
    filtered = filter_points(points, 50)
    filtered.sort(key=lambda point: point[1])
    return [pt for index,pt in zip(range(5), filtered)]

#This function returns the distance between two points.
def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (b[1] - a[1]) ** 2)

#This function filters the points that are stored in the list called points.
#This function filteres the points based on the distance between the points and the new,
#(distance between the points is atleast 50 in this case).
#points are stored in new list called filtered
def filter_points(points, filtervalue):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i] and points[j] and dist(points[i], points[j]) < filtervalue:
                points[j] = None
    filtered = []
    for point in points:
        if point is not None:
            filtered.append(point)
    return filtered

#This function draws small circle at the tip of fingers.
def plot(frame, points):
    radius = 5
    color = (0, 0, 255)
    thickness = -1
    for point in points:
        cv2.circle(frame, point, radius, color, thickness)

cap=cv2.VideoCapture(0)
screen=np.zeros((1000,600))
hist=capture_histogram(0)
curr=None
prev=None
while cap.isOpened():
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(1000,1000))
    hand_detected,hand=detect_hand(frame,hist)
    if hand_detected:
        hand_image=hand['boundaries']
        finger_tips=extract_finger_tips(hand)
        plot(hand_image,finger_tips)
        prev=curr
        curr=finger_tips[0]
        #print(curr)
        if prev and curr:
            cv2.line(screen,prev,curr,(134,12,234),3)
        cv2.imshow('drawing',screen)
        cv2.imshow('hand_detecter',hand_image)
    else:
        cv2.imshow('hand_detector',frame)
    if cv2.waitKey(1) & 0XFF==ord('x'):
        break


cap.release()
cv2.destroyAllWindows()
