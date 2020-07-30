# contours: continuous lines or curves the bound the object


import cv2
import numpy as np

# load the image

image = cv2.imread('../images/sudoku.jpg')

# convert it to grayscale

image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("original image", image)
cv2.waitKey(0)

# find the canny edges

edged = cv2.Canny(image,30,200)
cv2.imshow("canny edged", edged)
cv2.waitKey(0)

# finding contours

copied = edged.copy()

contours, hierarchy = cv2.findContours(copied,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.imshow("canny edge after contouring", copied)
cv2.waitKey(0)

print(" NUMBER OF contour found "+ str(len(contours)))

# draw all contours

cv2.drawContours(image, contours, -1, (0,255,0), 3)

cv2.imshow("contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()