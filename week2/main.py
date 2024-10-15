import cv2
import numpy as np
import poisson_editing
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import sys

def read_image(path):
    return plt.imread(path)

def read_mask(path):
    img = plt.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 0

def plot_clonning_goal(dst_image, transplants):
    plt
    pass

def poisson_cloning(dst, transplants):
    transplant = np.zeros_like(dst)
    mask = np.zeros(shape=dst.shape[:2], dtype=bool)
    for src_image, src_mask, dst_mask in transplants:
        t = poisson_editing.get_transplant(src_image, src_mask, dst_mask)
        transplant[t>0] = t[t>0]
        mask |= dst_mask
    result = poisson_editing.simple_poisson_solver(dst.astype(np.float32), 
                                                   transplant.astype(np.float32), 
                                                   mask.astype(np.float32))
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

if sys.argv[1] == 'lena':
    dst = read_image('images/lena/lena.png')
    src = read_image('images/lena/girl.png')
    print(dst.dtype, src.dtype)
    src_mask_eyes = read_mask('images/lena/mask_src_eyes.png')
    dst_mask_eyes = read_mask('images/lena/mask_dst_eyes.png')
    src_mask_mouth = read_mask('images/lena/mask_src_mouth.png')
    dst_mask_mouth = read_mask('images/lena/mask_dst_mouth.png')
    transplants = [
        (src, src_mask_eyes, dst_mask_eyes),
        (src, src_mask_mouth, dst_mask_mouth)
    ]
    cloning_result = poisson_cloning(dst, transplants)

if sys.argv[1] == 'mona':
    dst = read_image('images/monalisa/monalisa.png')
    src = read_image('images/monalisa/ginevra.png')
    src_mask = read_mask('images/monalisa/mask.png')
    dst_mask = read_mask('images/monalisa/mask.png')

    transplants = [
        (src, src_mask, dst_mask)
    ]
    cloning_result = poisson_cloning(dst, transplants)




#     # poisson_cloning
# # Load images
# src = read_image('images/lena/girl.png')
# dst = read_image('images/lena/lena.png')

# # For Mona Lisa and Ginevra:
# # src = cv2.imread('images/monalisa/ginevra.png')
# # dst = cv2.imread('images/monalisa/monalisa.png')

# # Customize the code with your own pictures and masks.

# # Store shapes and number of channels (src, dst and mask should have same dimensions!)
# ni, nj, nChannels = dst.shape

# # # Display the images
# # cv2.imshow('Source image', src); cv2.waitKey(0)
# # cv2.imshow('Destination image', dst); cv2.waitKey(0)

# # Load masks for eye swapping
# src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
# dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)
# # cv2.imshow('Eyes source mask', src_mask_eyes); cv2.waitKey(0)
# # cv2.imshow('Eyes destination mask', dst_mask_eyes); cv2.waitKey(0)

# # Load masks for mouth swapping
# src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
# dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)
# # cv2.imshow('Mouth source mask', src_mask_mouth); cv2.waitKey(0)
# # cv2.imshow('Mouth destination mask', dst_mask_mouth); cv2.waitKey(0)

# # Get the translation vectors (hard coded)
# # t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes)
# # t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth)

# # Cut out the relevant parts from the source image and shift them into the right position
# # Blend with the original (destination) image
# # CODE TO COMPLETE

# combined = poisson_editing.combine_images(src, dst, src_mask_eyes, dst_mask_eyes)
# combined = poisson_editing.combine_images(src, combined, src_mask_mouth, dst_mask_mouth)
# plt.imshow(combined)
# plt.show()

# # # GRADIENT TEST
# # gi, gj = poisson_editing.im_fwd_gradient(combined[:,:,0])

# # # DIVERGENCE TEST
# # div = poisson_editing.im_bwd_divergence(gi, gj)

# mask = np.zeros_like(dst)
# u_comb = combined # combined image

# mask = dst_mask_eyes | dst_mask_mouth
# u_final = np.zeros_like(dst)

# for channel in range(3):

#     m = mask[:, :, channel] > 0
#     u = u_comb[:, :, channel].astype(np.float64)
#     f_star = dst[:, :, channel].astype(np.float64)
#     # u1 = src[:, :, channel]

#     beta_0 = 1   # TRY CHANGING
#     beta = beta_0 * (1 - mask)

#     A, b = poisson_editing.poisson_linear_operator(f_star, u, m, beta)
#     x = spsolve(A, b)
#     img = np.clip(np.reshape(x, f_star.shape), 0, 255).astype(int)
#     plt.imshow(img, cmap='gray')
#     plt.show()
#     u_final[:,:, channel] = img
# plt.imshow(u_final)
# plt.show()