# finding the convex hall

# smallest length that can bound the perimeter

import numpy as np
import cv2


# load the image

image = cv2.imread('../images/hand.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cv2.imshow("original image", image)

# perform thresholding

ret, thresh = cv2.threshold(gray_image,127, 255, cv2.THRESH_BINARY_INV)

# find the contours

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# sort contour by area and remove the largest one

# n = len(contours)-1
# contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

# iterate through contour and draw the convex hall

for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull],0, (0,255,0),2)
    cv2.imshow("convex hull", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
