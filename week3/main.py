import cv2
import numpy as np
import segmentation
import sys
from tqdm import tqdm

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
cv2.imshow('Normalized image',img); cv2.waitKey(0)

# Height and width
ni = img.shape[0]
nj = img.shape[1]

# Make color images grayscale. Skip this block if you handle the multi-channel Chan-Sandberg-Vese model
if len(img.shape) > 2:
    nc = img.shape[2] # number of channels
    img = np.mean(img, axis=2)

# Try out different parameters
type_opt = "gradient-descent"
mu = 1
nu = 1
lambda1 = 1
lambda2 = 1
tol = 0.1
dt = (1e-2)/mu
iterMax = 50001

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

# Segmented image
seg = np.zeros(shape=img.shape)
seg[phi >= 0] = 1.0
seg[phi < 0] = 0.0

# Show output image
# cv2.imshow('Segmented image', seg)

# CODE TO COMPLETE
# Explicit gradient descent or Semi-explicit (Gauss-Seidel) gradient descent (Bonus)
# Extra: Implement the Chan-Sandberg-Vese model (for colored images)
# Refer to Getreuer's paper (2012)

# CODE TO COMPLETE
epsilon = 1.0
if type_opt == "gradient-descent":
    print("Iterating...")
    #only works for gray level img
    for iter in tqdm(range(int(iterMax))):
        # compute/update c1 and c2
        c1,c2 = segmentation.get_level_set_averages(img, phi)
        # new iteration
        direction = segmentation._level_set_gradient(phi, img, c1, c2, mu, nu, lambda1, lambda2, epsilon=epsilon)
        # update phi
        phi = phi - direction*dt


        # if np.linalg.norm(new_phi-phi)/np.sum(phi>=0) <= tol:
        #     #end cond to avoid to do all the iteration if the changes are smaller 
        #     break
elif type_opt == "gauss-seidel":
    print("Iterating with Gauss-Seidel...")
    for iter in tqdm(range(int(iterMax))):
        c1, c2 = segmentation.get_level_set_averages(img, phi)
        phi = segmentation._level_set_gauss_seidel(phi, img, c1, c2, mu, nu, lambda1, lambda2, epsilon, dt)
        

        

elif type_opt == "gauss-seidel-color":
    for iter in range(int(iterMax)):
        # heaviside function H(phi(x))
        H_phi_x = 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))

        # compute c1 and c2
        c1 = np.sum(H_phi_x * img) / (np.sum(H_phi_x) + 1e-10) #tiny number to avoid zero division
        c2 = np.sum((1 - H_phi_x) * img) / (np.sum(1 - H_phi_x) + 1e-10)
else:
    sys.exit("Error wrong method")

# Segmented image
seg = np.zeros(shape=img.shape)

# CODE TO COMPLETE
seg[phi >= 0] = 1.0
seg[phi < 0] = 0.0

# Show output image
cv2.imshow('Segmented image', seg)
cv2.waitKey(0)