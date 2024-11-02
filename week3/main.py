import cv2
import numpy as np

folderInput = 'images/'
figure_name = 'circles.png'
figure_name_final = folderInput + figure_name
img = cv2.imread(figure_name_final, cv2.IMREAD_UNCHANGED)
img = img.astype('float')

# Visualize the grayscale image
cv2.imshow('Image', img); cv2.waitKey(0)

# Normalize image
img = (img.astype('float') - np.min(img))
img = img/np.max(img)
cv2.imshow('Normalized image',I)
cv2.waitKey(0)

# Height and width
ni = img.shape[0]
nj = img.shape[1]

# Make color images grayscale. Skip this block if you handle the multi-channel Chan-Sandberg-Vese model
if len(img.shape) > 2:
    nc = img.shape[2] # number of channels
    img = np.mean(img, axis=2)

# Try out different parameters
mu = 1
nu = 1
lambda1 = 1
lambda2 = 1
tol = 0.1
dt = (1e-2)/mu
iterMax = 1e5

X, Y = np.meshgrid(np.arange(0, nj), np.arange(0, ni), indexing='xy')

# Initial phi
# This initialization allows a faster convergence for phantom2
phi = (-np.sqrt((X - np.round(ni / 2)) ** 2 + (Y - np.round(nj / 2)) ** 2) + 50)
# Alternatively, you may initialize randomly, or use the checkerboard pattern as suggested in Getreuer's paper

# Normalization of the initial phi to the range [-1, 1]
min_val = np.min(phi)
max_val = np.max(phi)
phi = phi - min_val
phi = 2 * phi / max_val
phi = phi - 1

# CODE TO COMPLETE
# Explicit gradient descent or Semi-explicit (Gauss-Seidel) gradient descent (Bonus)
# Extra: Implement the Chan-Sandberg-Vese model (for colored images)
# Refer to Getreuer's paper (2012)

# Segmented image
seg = np.zeros(shape=img.shape)

# CODE TO COMPLETE

# Show output image
cv2.imshow('Segmented image', seg); cv2.waitKey(0)