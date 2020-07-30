

import numpy as np
import cv2

# load the image

image = cv2.imread('../images/blobs.png',0)
cv2.imshow("original image", image)
cv2.waitKey(0)

# initialize the detector using the default parameter

detector = cv2.SimpleBlobDetector()

# detect blobs

keypoints = detector.detect(image)

# draw blobs on our image as red circles

blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

num_of_blobs = len(keypoints)
text = " total number of blobs is"+str(num_of_blobs)
cv2.putText(blobs,text,(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(100,0,255),2)

# display the image with blobs keypoints
cv2.imshow("Blobs using Defualt Parameter", blobs)
cv2.waitKey(0)


# set our filtering parameters

params = cv2.SimpleBlobDetector_Params()

# set area filtering parameters
params.filterByArea = True
params.minArea = 100

# set circularity filtering parameters
params.filterByCircularity = True
params.minCircularity =0.9

# set convexity filtering parameters
params.filterByConvexity = False
params.minConvexity=0.2

# set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio=0.01

# create a detector with these parameters

detector = cv2.SimpleBlobDetector(params)

# DETECT blobs

keypoints = detector.detect(image)

# draw blobs on our image as red circles

blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image,keypoints,blank,(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

num_of_blobs = len(keypoints)
text = " number of circular blobs is:"+str(num_of_blobs)
cv2.putText(blobs, text, (20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

# SHOW BLOBS

cv2.imshow("filtering circular blobs only",blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
