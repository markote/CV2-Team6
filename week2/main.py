import cv2
import numpy as np
import poisson_editing

# Load images
src = cv2.imread('images/lena/girl.png')
dst = cv2.imread('images/lena/lena.png')
# For Mona Lisa and Ginevra:
# src = cv2.imread('images/monalisa/ginevra.png')
# dst = cv2.imread('images/monalisa/monalisa.png')

# Customize the code with your own pictures and masks.

# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# Display the images
cv2.imshow('Source image', src); cv2.waitKey(0)
cv2.imshow('Destination image', dst); cv2.waitKey(0)

# Load masks for eye swapping
src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)
cv2.imshow('Eyes source mask', src_mask_eyes); cv2.waitKey(0)
cv2.imshow('Eyes destination mask', dst_mask_eyes); cv2.waitKey(0)

# Load masks for mouth swapping
src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)
cv2.imshow('Mouth source mask', src_mask_mouth); cv2.waitKey(0)
cv2.imshow('Mouth destination mask', dst_mask_mouth); cv2.waitKey(0)

# Get the translation vectors (hard coded)
t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes, "eyes")
t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth, "mouth")

# Cut out the relevant parts from the source image and shift them into the right position
# CODE TO COMPLETE

# Blend with the original (destination) image
# CODE TO COMPLETE
mask = np.zeros_like(dst)
u_comb = np.zeros_like(dst) # combined image

for channel in range(3):

    m = mask[:, :, channel]
    u = u_comb[:, :, channel]
    f = dst[:, :, channel]
    u1 = src[:, :, channel]

    beta_0 = 1   # TRY CHANGING
    beta = beta_0 * (1 - mask)

    vi, vj = poisson_editing.composite_gradients(u1, f, mask)
    b = 0 # CODE TO COMPLETE

    u_final = 0 # CODE TO COMPLETE (e.g., using a scipy solver, or similar)

cv2.imshow('Final result of Poisson blending', u_final)