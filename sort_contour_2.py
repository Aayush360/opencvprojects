
import numpy as np
import cv2

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

# function to calculate contour area and return as a list

def get_contour_area(contours):
    all_areas=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


# print the contours area before sorting

print("contour area before sorting: ", get_contour_area(contours))


# sort contours large to small

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# print contour area after sorting

print(" contour area after sorting ", get_contour_area(sorted_contours))

# iterate over a contour and draw one at a time

for c in sorted_contours:
    cv2.drawContours(original_image,[c],-1,(255,0,0),2)
    cv2.waitKey(0)
    cv2.imshow("contour by area", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

