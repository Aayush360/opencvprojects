
import cv2
import numpy as np

# load the image

image = cv2.imread('../images/sudoku.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# HarrisCorner funtion requires array datatype to be float
gray = np.float32(gray)

harris_corner = cv2.cornerHarris(gray,3,3,0.05)

# we use dilation of corner points to enlarge them

kernel = np.ones((7,7), np.uint8)
harris_corner = cv2.dilate(harris_corner,kernel, iterations=2)

# threshold for an optimal value, may vary depending on the image

image[harris_corner>0.025*harris_corner.max()] =[255,127,127] # to change the color of the corner

cv2.imshow("Harris Corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()