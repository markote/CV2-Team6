import os
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
    if len(img.shape) > 2:
        if img.shape[2] == 4:
            img = img[:,:,:3]
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
        plt.axis('off')
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

def poisson_cloning(dst, transplants, data_fidelity, mix=False):
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
    for i in range(3):
        result[:,:,i] = poisson_editing.poisson_solver(dst[:,:,i].astype(np.float32), 
                                                   transplant[:,:,i].astype(np.float32), 
                                                   mask, beta=data_fidelity, mix_type=mix)
    result = np.clip(result, 0, 1)
    bad_clonning = np.copy(dst)
    bad_clonning[mask] = transplant[mask]
    return result, bad_clonning

if sys.argv[1] == 'lena':
    data_fidelity = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    mix = sys.argv[3] if len(sys.argv) >= 4 else ''
    print(data_fidelity)
    print(mix)
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
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, data_fidelity, mix)
    plot_comparison(cloning_raw, cloning_result)

if sys.argv[1] == 'mona':
    data_fidelity = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    mix = sys.argv[3] if len(sys.argv) >= 4 else ''
    print(mix)
    dst = read_image('images/monalisa/lisa.png')
    src = read_image('images/monalisa/ginevra.png')
    src_mask = read_mask('images/monalisa/mask.png')
    dst_mask = read_mask('images/monalisa/mask.png')

    transplants = [
        (src, src_mask, dst_mask)
    ]
    plot_clonning_goal(dst, transplants)
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, data_fidelity, mix)
    plot_comparison(cloning_raw, cloning_result)

if sys.argv[1] == 'ivan':
    data_fidelity = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    mix = sys.argv[3] if len(sys.argv) >= 4 else ''
    print(mix)
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
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, data_fidelity, mix)
    plot_comparison(cloning_raw, cloning_result)

if sys.argv[1] == 'ivan_batch':
    data_fidelity_values = np.linspace(0, 1, 6)

    src = read_image('images/sandler/ivan.jpg')
    dst = read_image('images/sandler/adam-sandler.jpg')
    plt.figure(figsize=(20, 10))

    plt.subplot(4, 2, 1)
    plt.imshow(src)
    plt.title('Source Image')
    plt.axis('off')

    plt.subplot(4, 2, 2)
    plt.imshow(dst)
    plt.title('Destination Image')
    plt.axis('off')

    for idx, data_fidelity in enumerate(data_fidelity_values):
        print(f"Beta: {data_fidelity}")

        src = read_image('images/sandler/ivan.jpg').astype(np.float32) / 255
        dst = read_image('images/sandler/adam-sandler.jpg').astype(np.float32) / 255
        dst_mask = read_mask('images/sandler/adam_mask.png')
        src_mask = read_mask('images/sandler/ivan_mask.png')

        src = src[: -269, : -12]
        print("Dst Image Type:", dst.dtype, "Src Image Type:", src.dtype)
        transplants = [(src, src_mask, dst_mask)]
        cloning_result, _ = poisson_cloning(dst, transplants, data_fidelity, False)
        result_save_path = os.path.join('', f"cloning_result_fidelity_{data_fidelity:.2f}.png")
        plt.imsave(result_save_path, cloning_result)
        plt.subplot(4, 2, idx + 3)
        plt.imshow(cloning_result)
        plt.title(f'Beta: {data_fidelity:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if sys.argv[1] == 'expert':
    data_fidelity = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    mix = sys.argv[3] if len(sys.argv) >= 4 else ''
    print(mix)
    dst = read_image('images/expert/theexpert.jpg').astype(np.float32) / 255
    src = read_image('images/expert/realexpert.jpg').astype(np.float32) / 255
    src_mask = read_mask('images/expert/realexpert_mask.png')
    dst_mask = read_mask('images/expert/theexpert_mask.png')
    # src = src[:,::-1]
    # src_mask = src_mask[:,::-1]
    print(dst.dtype, src.dtype, np.max(dst))
    transplants = [
        (src, src_mask, dst_mask)
    ]
    plot_clonning_goal(dst, transplants)
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, data_fidelity, mix)
    plot_comparison(cloning_raw, cloning_result)

if sys.argv[1] == 'coral':
    data_fidelity = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    mix = sys.argv[3] if len(sys.argv) >= 4 else ''
    print(mix)
    dst = read_image('images/coral/coral_reef.jpeg').astype(np.float32) / 255
    src = read_image('images/coral/phat.jpg').astype(np.float32) / 255
    src_mask = read_mask('images/coral/phat_mask.png')
    dst_mask = read_mask('images/coral/phat_mask.png')

    transplants = [
        (src, src_mask, dst_mask)
    ]
    plot_clonning_goal(dst, transplants)
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, 0, '')
    i = 1
    plt.imsave('phat_coral'+str(i)+'.png', cloning_result)

    i+=1
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, 0.15, '')
    plt.imsave('phat_coral'+str(i)+'.png', cloning_result)

    i+=1
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, 0, 'mean')
    plt.imsave('phat_coral'+str(i)+'.png', cloning_result)

    i+=1
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, 0.1, 'mean')
    plt.imsave('phat_coral'+str(i)+'.png', cloning_result)

if sys.argv[1] == 'mugshot':
    data_fidelity = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    mix = sys.argv[3] if len(sys.argv) >= 4 else ''
    print(mix)
    dst = read_image('images/mugshot/marco.jpg').astype(np.float32) / 255
    src = read_image('images/mugshot/mugshot.jpg').astype(np.float32) / 255
    src_mask = read_mask('images/mugshot/mugshot_mask.png')
    dst_mask = read_mask('images/mugshot/marco_mask.png')
    # src = src[:,::-1]
    # src_mask = src_mask[:,::-1]
    transplants = [
        (src, src_mask, dst_mask)
    ]
    plot_clonning_goal(dst, transplants)
    cloning_result, cloning_raw = poisson_cloning(dst, transplants, data_fidelity, mix)
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