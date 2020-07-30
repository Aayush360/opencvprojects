import cv2
import numpy as np

# load the image, grayscale it and threshold

image = cv2.imread('../images/shape2.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(image_gray,127,255,1)


# extract contours

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    # get approximate polygons
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True),True)


    if len(approx)==3:
        shape_name = " Triangle"
        cv2.drawContours(image, [cnt],0,(0,255,0),-1)

        # find contour center to place text at the center
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(image,shape_name,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)

    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # check if it is square or rectangle
        if abs(w-h)<=3:
            shape_name ="Square"

            # find contour center to place the text at the center

            cv2.drawContours(image,[cnt],0,(0,255,122),-1)
            cv2.putText(image, shape_name, (cx-50,cy), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0), 2)
        else:
            shape_name ="rectangle"

            cv2.drawContours(image,[cnt],0,(0,125,255),-1)
            cv2.putText(image,shape_name,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

    elif len(approx) == 6:
        shape_name="hexagon"
        cv2.drawContours(image,[cnt],0,(0,255,255),-1)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(image,shape_name,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

    elif len(approx) >= 15:
        shape_name="circle"
        cv2.drawContours(image,[cnt],0,(0,255,255,-1))
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(image,shape_name,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)


cv2.imshow("Identifying shape", image)
cv2.waitKey(0)

cv2.destroyAllWindows()






