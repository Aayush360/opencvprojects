import cv2
import numpy as np


# load the image and convert to grayscale

image = cv2.imread('../images/waldo.jpg')
cv2.imshow("original waldo ", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# load the template

template = cv2.imread('../images/waldo_template.png',0)

# matching template

result = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# creating bounding box

top_left = max_loc
bottom_right = (top_left[0]+50, top_left[1]+50)
cv2.rectangle(image,top_left,bottom_right,(0,255,0),2)

cv2.imshow("Where is Waldo? ", image)
cv2.waitKey(0)
cv2.destroyAllWindows()