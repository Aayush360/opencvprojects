
import cv2
import numpy as np


# load the image both target and template

template = cv2.imread('../images/star.jpg',0)
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# load the target image

target = cv2.imread('../images/shapetobematched.png')
target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)


# binarize both the terget and template image

ret1, thresh_target = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY_INV)
ret2, thresh_template = cv2.threshold(template,127,255,0)

# find the contour in template

contours, hierarchy = cv2.findContours(thresh_template,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# let us sort the contour by area to remove the largest contour which is the image outline

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# extract second larges contour which is our template contour

template_contour = contours[0]

# extract contour from second target image

contours, hierarchy = cv2.findContours(thresh_target, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # iterate through each contour in the target image
    # and use cv2.matchShape to find the matching shape to template in target

    match = cv2.matchShapes(template_contour,c, 1,0.0)
    # lower value means closer match to the original image
    print(match)
    if match<0.15:
        closest_contour = c
    else:
        closest_contour= []

closest_contour = np.array(closest_contour)
cv2.drawContours(target,[closest_contour],-1, (0,255,0),3)
cv2.imshow("output", target)
cv2.waitKey(0)
cv2.destroyAllWindows()
