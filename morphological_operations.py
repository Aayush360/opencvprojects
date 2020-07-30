
# dialation, erosion, opening and closing
# dilation: add pixels (white) to the boundaries of the pixel
# erosion: removes pixels from the boundaries of the pixel
# closing: dilation followed by erosion
# opening: erosion followed by dilation

import cv2
import numpy as np


image = cv2.imread('../images/note.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
x, image = cv2.threshold(image,100,255,cv2.THRESH_BINARY_INV)

print("threshold value is", x)

cv2.imshow("binary image", image)

cv2.waitKey(0)


# let us define our kernel size

kernel = np.ones((5,5), dtype='uint8')

# now we erode

erosion = cv2.erode(image, kernel, iterations=1)
cv2.imshow("eroded image", erosion)
cv2.waitKey(0)

# dialation

dilation = cv2.dilate(image, kernel, iterations=1)
cv2.imshow("dilation", dilation)
cv2.waitKey(0)

# opening - good for removing noise

opening = cv2.morphologyEx(image,cv2.MORPH_OPEN, kernel)
cv2.imshow("opening", opening)
cv2.waitKey(0)

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closing", closing)
cv2.waitKey(0)
