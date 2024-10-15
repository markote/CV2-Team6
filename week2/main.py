import cv2
import numpy as np
import poisson_editing
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from skimage import color
import sys

def read_image(path):
    img = plt.imread(path)
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img

def read_mask(path):
    img = plt.imread(path)
    if img.ndim == 3:
        return color.rgb2gray(img) > 0
    return img > 0

def make_translucid_mask(mask):
    result = np.zeros(shape=(*mask.shape, 4), dtype=np.float32)
    result[:,:,0] = 1
    result[:,:,1] = 1
    result[:,:,2] = 1
    result[:,:,3] = mask*0.5
    return result

def plot_clonning_goal(dst_image, transplants):
    r = len(transplants)
    _, ax = plt.subplots(r, 2, figsize=(6, 3*r), squeeze=False)
    for i in range(len(transplants)):
        ax[i][0].imshow(dst_image)
        ax[i][0].imshow(make_translucid_mask(transplants[i][2]))
        ax[i][1].imshow(transplants[i][0])
        ax[i][1].imshow(make_translucid_mask(transplants[i][1]))
    plt.show()

def plot_comparison(img1, img2):
    _, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    plt.show()

def poisson_cloning(dst, transplants, data_fidelity):
    transplant = np.zeros_like(dst)
    print(transplant.shape)
    mask = np.zeros(shape=dst.shape[:2], dtype=bool)
    for src_image, src_mask, dst_mask in transplants:
        print(dst_mask.shape)
        t = poisson_editing.get_transplant(src_image, src_mask, dst_mask)
        print(t.shape)
        transplant[t>0] = t[t>0]
        mask |= dst_mask
    result = np.zeros_like(dst)
    plt.imshow(transplant)
    plt.show()
    plt.imshow(dst)
    plt.show()
    for i in range(3):
        result[:,:,i] = poisson_editing.poisson_solver(dst[:,:,i].astype(np.float32), 
                                                   transplant[:,:,i].astype(np.float32), 
                                                   mask, beta=data_fidelity)
    result = np.clip(result, 0, 1)
    bad_clonning = np.copy(dst)
    bad_clonning[mask] = transplant[mask]
    return result, bad_clonning

if sys.argv[1] == 'lena':
    data_fidelity = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    print(data_fidelity)
    dst = read_image('images/lena/lena.png')
    src = read_image('images/lena/girl.png')
    print(dst.dtype, src.dtype)
    print(dst.shape, src.shape)
    src_mask_eyes = read_mask('images/lena/mask_src_eyes.png')
    dst_mask_eyes = read_mask('images/lena/mask_dst_eyes.png')
    src_mask_mouth = read_mask('images/lena/mask_src_mouth.png')
    dst_mask_mouth = read_mask('images/lena/mask_dst_mouth.png')
    transplants = [
        (src, src_mask_eyes, dst_mask_eyes),
        (src, src_mask_mouth, dst_mask_mouth)
    ]
    plot_clonning_goal(dst, transplants)
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, data_fidelity)
    plot_comparison(cloning_raw, cloning_result)

if sys.argv[1] == 'mona':
    data_fidelity = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    dst = read_image('images/monalisa/lisa.png')
    src = read_image('images/monalisa/ginevra.png')
    src_mask = read_mask('images/monalisa/mask.png')
    dst_mask = read_mask('images/monalisa/mask.png')

    transplants = [
        (src, src_mask, dst_mask)
    ]
    plot_clonning_goal(dst, transplants)
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, data_fidelity)
    plot_comparison(cloning_raw, cloning_result)

if sys.argv[1] == 'ivan':
    data_fidelity = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    dst = read_image('images/sandler/ivan.jpg').astype(np.float32) / 255
    src = read_image('images/sandler/adam-sandler.jpg').astype(np.float32) / 255
    src_mask = read_mask('images/sandler/adam_mask.png')
    dst_mask = read_mask('images/sandler/ivan_mask.png')
    dst = dst[: -269, : -12]
    print(dst.dtype, src.dtype)

    transplants = [
        (src, src_mask, dst_mask)
    ]
    plot_clonning_goal(dst, transplants)
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, data_fidelity)
    plot_comparison(cloning_raw, cloning_result)




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