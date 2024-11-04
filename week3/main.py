import cv2
import numpy as np
from scipy.ndimage import generic_filter
from segmentation import initialize_level_set, _level_set_gradient, compute_binaryvalue_c1c2
import sys

# default parameters
mu = 1
nu = 0.045
lambda1 = 1
lambda2 = 1
tol = 1e-4
dt = 0.5
iterMax = 3e4

if sys.argv[1] == 'phantom3':
    folderInput = 'images/'
    figure_name = 'phantom3.bmp'
    figure_name_final = folderInput + figure_name
    img = cv2.imread(figure_name_final, cv2.IMREAD_UNCHANGED)
    img = generic_filter(img, np.var, size=11)
    img = cv2.GaussianBlur(img,(11,11),0)

    mu = 1
    nu = 0.045
    lambda1 = 1
    lambda2 = 1
    tol = 1e-4
    dt = 0.5
    iterMax = 3e4
    epsilon = 1.5
elif sys.argv[1] == 'phantom2':
    folderInput = 'images/'
    figure_name = f'{sys.argv[1]}.bmp'
    figure_name_final = folderInput + figure_name
    img = cv2.imread(figure_name_final, cv2.IMREAD_UNCHANGED)
    img = cv2.bilateralFilter(img,11,75,75)
    epsilon = 1.5
elif sys.argv[1] == 'phantom1':
    folderInput = 'images/'
    figure_name = f'{sys.argv[1]}.bmp'
    figure_name_final = folderInput + figure_name
    img = cv2.imread(figure_name_final, cv2.IMREAD_UNCHANGED)
    mu = 0.2
    nu = 0.001
    epsilon = 1
else: 
    folderInput = 'images/'
    figure_name = f'circles.png'
    figure_name_final = folderInput + figure_name
    img = cv2.imread(figure_name_final, cv2.IMREAD_UNCHANGED)
    mu = 0.2
    nu = 0.001
    epsilon = 1
    
img = img.astype('float')

# Visualize the grayscale image
cv2.imshow('Image', img); cv2.waitKey(0)

# Normalize image
img = (img.astype('float') - np.min(img))
img = img/np.max(img)
cv2.imshow('Normalized image',img); cv2.waitKey(0)
# Height and width
ni = img.shape[0]
nj = img.shape[1]

# Make color images grayscale. Skip this block if you handle the multi-channel Chan-Sandberg-Vese model
if len(img.shape) > 2:
    nc = img.shape[2] # number of channels
    img = np.mean(img, axis=2)

# Initial phi
# This initialization allows a faster convergence for phantom2
phi = initialize_level_set((ni,nj))
# Alternatively, you may initialize randomly, or use the checkerboard pattern as suggested in Getreuer's paper

# CODE TO COMPLETE
# Explicit gradient descent or Semi-explicit (Gauss-Seidel) gradient descent (Bonus)
# Extra: Implement the Chan-Sandberg-Vese model (for colored images)
# Refer to Getreuer's paper (2012)


for iter in range(int(iterMax)):
    c1,c2 = compute_binaryvalue_c1c2(f= img, phi= phi, epsilon= epsilon)
    dphi_dt = _level_set_gradient(phi, img, c1, c2,mu,nu,lambda1,lambda2,epsilon)

    # update phi
    phi_old = phi.copy()
    phi = phi + dt * dphi_dt

    # converge
    if np.max(np.abs(phi - phi_old)) < tol:
        print('Converged at iteration', iter)
        break
    # display phi
    if iter % 1000 == 0:
        print('Iteration:', iter)
        phi_display = (phi - np.min(phi)) / (np.max(phi) - np.min(phi))
        phi_display = (phi_display * 255).astype(np.uint8)
        cv2.imshow('Phi', phi_display)
        cv2.waitKey(1)

# Segmented image
seg = np.zeros_like(img)
seg[phi >= 0] = 1
seg = (seg * 255).astype(np.uint8)

# Show output image
cv2.imshow('Segmented image', seg); cv2.waitKey(0)