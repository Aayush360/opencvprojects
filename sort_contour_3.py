# sorting contour from left to right (by position)

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

def x_cord_contour(contours):
    if cv2.contourArea(contours) >10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))


def label_contour_center(image, c,i):
    # places a red circle at the center of the contours
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # draw the contour number on the image

    cv2.circle(image,(cx,cy),10,(0,0,255),-1)
    return image


# load the image

image = cv2.imread('../images/shapes.jpg')
original_image = image.copy()

# compute center of mass or centroid and compute them on our image

for (i,c) in enumerate(contours):
    orig = label_contour_center(image,c,i)

cv2.imshow(" contour center ", image)
cv2.waitKey(0)

# sort by left to right using x-cord-contour function

contours_left_to_right = sorted(contours, key= x_cord_contour, reverse= False)


# labeling contour from left to right

for (i,c) in enumerate(contours_left_to_right):
    cv2.drawContours(original_image,[c],-1,(0,255,0),3)
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.putText(original_image, str(i+1),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.imshow("left to right contour", original_image)
    cv2.waitKey(0)
    (x,y,w,h) = cv2.boundingRect(c)

    # let's now crop each contour and save these images
    cropped_contour = original_image[y:y+h, x:x+w]
    image_name = "output_shape_number_"+str(i+1)+".jpg"
    print(image_name)
    cv2.imwrite(image_name,cropped_contour)

cv2.destroyAllWindows()





