import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the image and convert it to gray scale

image = cv2.imread('../images/Lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# show original image

cv2.imshow("Original image", image)
cv2.waitKey(0)

# h x w in pixels

cell_size = (8,8)

# h x w in cell size

block_size = (2,2)

# number of orientation bins
n_bins = 9

# using opencv's HOG Descriptor
# window size is the size of the image cropped to multiple of cell size

hog = cv2.HOGDescriptor(_winSize=(gray.shape[1]//cell_size[1]* cell_size[1],
                                  gray.shape[0]//cell_size[0]*cell_size[0]),
                        _blockSize=(block_size[1]*cell_size[1],
                                    block_size[0]*cell_size[0]),
                        _cellStride = (cell_size[1],cell_size[0]),
                        _cellSize=(cell_size[1],cell_size[0]),
                        _nbins = n_bins)
# create numpy array shape which we use to create hough features

n_cells = (gray.shape[0]//cell_size[0], gray.shape[1]//cell_size[1])

# we index blocks by row first
# hog_feats now contain gradient amplitudes for each direction
# for each cell of its group, for each group. Indexing is by rows and then columns

hog_feats = hog.compute(gray).reshape(n_cells[1]-block_size[1]+1,
                                      n_cells[0]-block_size[0]+1,
                                      block_size[0],block_size[1],n_bins).transpose((1,0,2,3,4))

# create our gradient array with nbin dimension to store gradeint orientation

gradients = np.zeros((n_cells[0],n_cells[1],n_bins))

# create array of dimension

cell_count = np.zeros((n_cells[0],n_cells[1],1),dtype=int)


# block Normalization

for off_y in range(block_size[0]):
    for off_x in range(block_size[1]):
        gradients[off_y: n_cells[0]-block_size[0]+off_y+1,
        off_x:n_cells[1]-block_size[1]+off_x+1]+= hog_feats[:,:,off_y,off_x, :]
        cell_count[off_y: n_cells[0] - block_size[0] + off_y + 1,
        off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

# average gradient

gradients/=cell_count

# plot HOGs using Matplotlib
# angle is 360/n_bins * direction

color_bins = 5
plt.pcolor(gradients[:,:,color_bins])
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()
cv2.destroyAllWindows()






