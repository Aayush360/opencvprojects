
import numpy as np
import cv2

# load the image

image = cv2.imread('../images/sudoku.jpg')
orig_image = image.copy()

cv2.imshow('original image', orig_image)
cv2.waitKey(0)

# convert the image to grayscale

gray_scale = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_scale,127,255,cv2.THRESH_BINARY_INV)

# find the contours

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# iterate through each contour and find the bounding rectangle

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("bounding ractangle", orig_image)

cv2.waitKey(0)

# iterate through each contour and compute the approx contour

for c in contours:
    # calculate the accuracy as percentage of contour perimeter
    accuracy = 0.03* cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(image,[approx],0,(0,255,0),2)
    cv2.imshow("approx polydp image", image)


cv2.waitKey(0)
cv2.destroyAllWindows()
