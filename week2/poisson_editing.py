import numpy as np
from scipy.signal import correlate2d
from skimage.feature import match_template
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as nd

def im_fwd_gradient(image: np.ndarray):

    # CODE TO COMPLETE
    grad_i = 0
    grad_j = 0
    return grad_i, grad_j

def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray):

    # CODE TO COMPLETE
    div_i = 0
    div_j = 0
    return div_i + div_j

def composite_gradients(u1: np.array, u2: np.array, mask: np.array):
    """
    Creates a vector field v by combining the forward gradient of u1 and u2.
    For pixels where the mask is 1, the composite gradient v must coincide
    with the gradient of u1. When mask is 0, the composite gradient v must coincide
    with the gradient of u2.

    :return vi: composition of i components of gradients (vertical component)
    :return vj: composition of j components of gradients (horizontal component)
    """

    # CODE TO COMPLETE
    vi = 0
    vj = 0
    return vi, vj

def poisson_linear_operator(u: np.array, beta: np.array):
    """
    Implements the action of the matrix A in the quadratic energy associated
    to the Poisson editing problem.
    """
    Au = 0
    # CODE TO COMPLETE
    return Au

def get_translation(original_img: np.ndarray, translated_img: np.ndarray, part: str = ""):

    # For the eyes mask:
    # The top left pixel of the source mask is located at (x=115, y=101)
    # The top left pixel of the destination mask is located at (x=123, y=125)
    # This gives a translation vector of (dx=8, dy=24)

    # For the mouth mask:
    # The top left pixel of the source mask is located at (x=125, y=140)
    # The top left pixel of the destination mask is located at (x=132, y=173)
    # This gives a translation vector of (dx=7, dy=33)

    # The following shifts are hard coded:
    if part == "lena_eyes":
        return (24, 8)
    if part == "lena_mouth":
        return (33, 7)
    
    img1 = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)>0
    img2 = cv2.cvtColor(translated_img, cv2.COLOR_BGR2GRAY)>0
    correlation = correlate2d(img1, img2, mode='full')
    max_index = np.unravel_index(np.argmax(correlation), correlation.shape)
    translation = (max_index[0] - img2.shape[0] + 1, max_index[1] - img2.shape[1] + 1)
    print(translation)
    return translation

    # Here on could determine the shift vector programmatically,
    # given an original image/mask and its translated version.
    # Idea: using maximal cross-correlation (e.g., scipy.signal.correlate2d), or similar. This is too slow!!!!

def _bounding_box(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def combine_images(src_img: np.ndarray, dst_img: np.ndarray,
                    src_mask: np.ndarray, dst_mask: np.ndarray):
    """
    Combines the src_img into dst_img using the src_mask and dst_mask.
    """
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
    src_mask = cv2.cvtColor(src_mask, cv2.COLOR_RGB2GRAY) > 0
    dst_mask = cv2.cvtColor(dst_mask, cv2.COLOR_RGB2GRAY) > 0
    rmin, rmax, cmin, cmax = _bounding_box(dst_mask)
    dst_mask_clipped = dst_mask[rmin:rmax+1, cmin:cmax+1]
    corr = nd.correlate(src_mask.astype(np.float32), dst_mask_clipped.astype(np.float32))
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    mask_h, mask_w = dst_mask_clipped.shape
    src_region_x_start = max(0, x - mask_w // 2)
    src_region_y_start = max(0, y - mask_h // 2)
    src_region_x_end = src_region_x_start + mask_w
    src_region_y_end = src_region_y_start + mask_h
    masked_src_img = src_img[src_region_y_start:src_region_y_end, src_region_x_start:src_region_x_end]
    dst_img[rmin:rmax+1, cmin:cmax+1][dst_mask_clipped] = masked_src_img[dst_mask_clipped]
    return cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)