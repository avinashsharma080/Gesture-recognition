#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np
from sklearn.metrics import pairwise


# In[8]:


background = None 
accumulated_weight = 0.5

roi_top =50
roi_bottom = 400
roi_left = 400
roi_right = 800


# In[13]:


def cal_acc_avg(frame, accumulated_weight):
    
    global background
    
    if background is None:
        background = frame.copy().astype('float')
        return None 
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)


# In[14]:


def segment(frame,threshold=25):                                                    # threshold here is minimum threshold
    
    diff = cv2.absdiff(background.astype('uint8'),frame)
    
    ret, thresholded = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)
    
    image,contours,hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    else:
        hand_segment = max(contours,key= cv2.contourArea)
        
        return (thresholded,hand_segment)
    


# In[15]:


def count_finger(thresholded,hand_segment):
    
    convex_hull = cv2.convexHull(hand_segment)
    
    extreme_top    = tuple(convex_hull[convex_hull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(convex_hull[convex_hull[:, :, 1].argmax()][0])
    extreme_left   = tuple(convex_hull[convex_hull[:, :, 0].argmin()][0])
    extreme_right  = tuple(convex_hull[convex_hull[:, :, 0].argmax()][0])
    
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
    
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]
    
    
    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)
    
    
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    
    image , contours, hierarchy = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0

    # loop through the contours found
    for cnt in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        out_of_wrist = (cY + (cY*0.25)) > (y+h)
        
        limit_points = ((circumference * 0.25) > cnt.shape[0])

        
        if out_of_wrist and limit_points:
            count  += 1
            
    return count


# In[16]:


cap = cv2.VideoCapture(0)
num_frames = 0 

while True :
    
    ret,frame = cap.read()
    
    frame_cpy = frame.copy()
    
    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
    
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    if num_frames < 60:
        cal_acc_avg(gray,accumulated_weight)
        
        if num_frames < 59:
            cv2.putText(frame_cpy,'getting background plz wait',(200,300),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger_count',frame_cpy)
    else:
        
        hand = segment(gray)
        
        if hand is not None:
            
            thresholded , hand_segment  = hand
            
            cv2.drawContours(frame_cpy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)
            
            fingers = count_finger(thresholded,hand_segment)
            
            cv2.putText(frame_cpy,str(fingers),(70,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            
            cv2.imshow('Thresholded',thresholded)
            
    cv2.rectangle(frame_cpy, (roi_left, roi_top), (roi_right, roi_bottom), (0,255,0), 2)
    
    num_frames += 1
    
    cv2.imshow('Finger count',frame_cpy)
    
    key = cv2.waitKey(1) & 0xFF 
    
    if key == 27:
        break
    
    
cap.release()
cv2.destroyAllWindows()


# In[ ]:




