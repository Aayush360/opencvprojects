import cv2
import numpy as np


# load the image

image = cv2.imread('../images/shapes.jpg')
cv2.imshow("original image", image)

# create black image with same dim as original image

black_img = np.zeros((image.shape[0], image.shape[1],3))

# create a copy of our original image

original_image = image.copy()

# grayscale the image

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# find canny edge

canny = cv2.Canny(gray_img,50,200)
cv2.imshow("1- canny edge", canny)
cv2.waitKey(0)

# find contours and find how many were found

contours, hierarchy = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(" number of contours found", len(contours))

# draw all contours over black image

cv2.drawContours(black_img,contours,-1,(0,255,0),2)
cv2.imshow("contour over black background", black_img)
cv2.waitKey(0)

# draw all contours over original image

cv2.drawContours(image,contours,-1,(0,255,0),2)
cv2.imshow("contours over original image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()