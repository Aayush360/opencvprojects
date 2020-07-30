# emphasizes the edges in an image

# opposite of blurring

import cv2
import numpy as np

image = cv2.imread('../images/Lenna.png')
cv2.imshow("original image", image)

# create or sharpening kernel, we do not need to normalize since the value sums to one already

kernel_sharpening = np.array([[-1,-1,-1],
                              [-1,9,-1],
                              [-1,-1,-1]])
# applying kernel to input image

sharpened = cv2.filter2D(image,-1,kernel_sharpening)
cv2.imshow("image sharpening", sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()