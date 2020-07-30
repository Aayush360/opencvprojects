# image pyramiding: helps to rescaling the images
# either downscaling or upscaling


import cv2

image = cv2.imread('../images/nature.jpg')

larger = cv2.pyrUp(image)
smaller = cv2.pyrDown(image)

cv2.imshow("original image", image)
cv2.imshow("smaller", smaller)
cv2.imshow('larger', larger)

cv2.waitKey(0)
cv2.destroyAllWindows()