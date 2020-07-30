import cv2
import numpy as np


# read the image

image = cv2.imread('../images/sudoku.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# find the canny edge

canny = cv2.Canny(gray,100,200,apertureSize=3)

# minimum line length of 5 px and maximum gap between lines of 10 px

lines = cv2.HoughLinesP(canny,1,np.pi/180,50,5,10)
print(lines.shape)


for x1,y1,x2,y2 in lines[0]:
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)


cv2.imshow("probabilistic hough lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()