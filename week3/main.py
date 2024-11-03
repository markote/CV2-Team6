import cv2
import numpy as np

folderInput = 'images/'
figure_name = 'circles.png'
figure_name_final = folderInput + figure_name
img = cv2.imread(figure_name_final, cv2.IMREAD_UNCHANGED)
img = img.astype('float')

# Visualize the grayscale image
# cv2.imshow('Image', img); cv2.waitKey(0)

# Normalize image
img = (img.astype('float') - np.min(img))
img = img/np.max(img)
# cv2.imshow('Normalized image',img)
# cv2.waitKey(0)

# Height and width
ni = img.shape[0]
nj = img.shape[1]

# Make color images grayscale. Skip this block if you handle the multi-channel Chan-Sandberg-Vese model
if len(img.shape) > 2:
    nc = img.shape[2] # number of channels
    img = np.mean(img, axis=2)

# Try out different parameters
mu = 0.2
nu = 0
lambda1 = 1
lambda2 = 1
tol = 1e-2
dt = 0.5
iterMax = 1e5

X, Y = np.meshgrid(np.arange(0, nj), np.arange(0, ni), indexing='xy')

# Initial phi
# This initialization allows a faster convergence for phantom2
phi = (-np.sqrt((X - np.round(ni / 2)) ** 2 + (Y - np.round(nj / 2)) ** 2) + 50)
# print(X.shape, Y.shape)
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

# CODE TO COMPLETE
epsilon = 1.0

for iter in range(int(iterMax)):
    # heaviside function H(phi(x))
    H_phi_x = 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))

    # compute c1 and c2
    c1 = np.sum(H_phi_x * img) / (np.sum(H_phi_x) + 1e-10) #tiny number to avoid zero division
    c2 = np.sum((1 - H_phi_x) * img) / (np.sum(1 - H_phi_x) + 1e-10)

    # compute delta_phi_x
    delta_phi_x = (epsilon / (np.pi * (epsilon**2 + phi**2)))

    # compute gradients of phi
    phi_y, phi_x = np.gradient(phi)

    # compute gradient magnitude
    grad_phi_mag = np.sqrt(phi_x**2 + phi_y**2) + 1e-10  

    # normalized gradients
    nx = phi_x / grad_phi_mag
    ny = phi_y / grad_phi_mag

    # Compute curvature
    nxx = np.gradient(nx, axis=1)
    nyy = np.gradient(ny, axis=0)
    curvature = nxx + nyy
    dphi_dt = delta_phi_x * (mu * curvature - nu - lambda1 * (img - c1)**2 + lambda2 * (img - c2)**2)

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