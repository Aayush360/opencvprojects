# find the canny edge before finding the hough line


import cv2
import numpy as np

# load the image

image = cv2.imread('../images/sudoku.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray,100,170,apertureSize=3)

# run HoughLines using rho accuracy of 1px,
# theta accuracy of np.pi/180 which is 1 degree
# line threshold is set to 240, number of points in a line

lines = cv2.HoughLines(canny,1,np.pi/180,240)

# we iterate through each line and convert it to format
# required by cv2.lines (requires endpoints)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b= np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*(a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*(a))
    cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)


cv2.imshow("Hough lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()