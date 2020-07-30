# improvement over harris corner detection


import cv2
import numpy as np

# load the image

image = cv2.imread('../images/sudoku.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# we specify the top 50 corners

corners = cv2.goodFeaturesToTrack(gray,50,0.01,15)

# 0.01 -- minimum accepted quality of image corners


for corner in corners:
    x, y = corner[0]
    x = int(x)
    y = int(y)
    cv2.rectangle(image, (x-10,y-10),(x+10,y+10), (0,255,0), 2)


cv2.imshow("Corners found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()