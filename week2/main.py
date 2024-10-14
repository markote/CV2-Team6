import cv2
import numpy as np
import poisson_editing
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Load images
src = cv2.imread('images/lena/girl.png')
dst = cv2.imread('images/lena/lena.png')
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# For Mona Lisa and Ginevra:
# src = cv2.imread('images/monalisa/ginevra.png')
# dst = cv2.imread('images/monalisa/monalisa.png')

# Customize the code with your own pictures and masks.

# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# # Display the images
# cv2.imshow('Source image', src); cv2.waitKey(0)
# cv2.imshow('Destination image', dst); cv2.waitKey(0)

# Load masks for eye swapping
src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)
# cv2.imshow('Eyes source mask', src_mask_eyes); cv2.waitKey(0)
# cv2.imshow('Eyes destination mask', dst_mask_eyes); cv2.waitKey(0)

# Load masks for mouth swapping
src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)
# cv2.imshow('Mouth source mask', src_mask_mouth); cv2.waitKey(0)
# cv2.imshow('Mouth destination mask', dst_mask_mouth); cv2.waitKey(0)

# Get the translation vectors (hard coded)
# t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes)
# t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth)

# Cut out the relevant parts from the source image and shift them into the right position
# Blend with the original (destination) image
# CODE TO COMPLETE

combined = poisson_editing.combine_images(src, dst, src_mask_eyes, dst_mask_eyes)
combined = poisson_editing.combine_images(src, combined, src_mask_mouth, dst_mask_mouth)
plt.imshow(combined)
plt.show()

# # GRADIENT TEST
# gi, gj = poisson_editing.im_fwd_gradient(combined[:,:,0])

# # DIVERGENCE TEST
# div = poisson_editing.im_bwd_divergence(gi, gj)

mask = np.zeros_like(dst)
u_comb = combined # combined image

mask = dst_mask_eyes | dst_mask_mouth
u_final = np.zeros_like(dst)

for channel in range(3):

    m = mask[:, :, channel] > 0
    u = u_comb[:, :, channel].astype(np.float64)
    f_star = dst[:, :, channel].astype(np.float64)
    # u1 = src[:, :, channel]

    beta_0 = 1   # TRY CHANGING
    beta = beta_0 * (1 - mask)

    A, b = poisson_editing.poisson_linear_operator(f_star, u, m, beta)
    x = spsolve(A, b)
    img = np.clip(np.reshape(x, f_star.shape), 0, 255).astype(int)
    plt.imshow(img, cmap='gray')
    plt.show()
    u_final[:,:, channel] = img
plt.imshow(u_final)
plt.show()