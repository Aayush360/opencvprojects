# group of connected pixels that share a common property

import cv2
import numpy as np


# load the image

image = cv2.imread('../images/sunflower.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# set up the detector with default parameter

detector = cv2.SimpleBlobDetector()

# detect blobs

keypoints = detector.detect(gray)

# draw the detected blobs as red circles
# DRAW_MATCHES_FLAGS_DEFAULT ensures size of the circle corresponding to the size of the blobs


blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(gray,keypoints,blank,(0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# show keypoints

cv2.imshow("blobs: ", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

