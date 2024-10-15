import numpy as np
from scipy.signal import correlate2d
from skimage.feature import match_template
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.sparse as sparse
import scipy.ndimage as nd
from scipy.sparse.linalg import spsolve

def _bounding_box(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def get_transplant(src_img: np.ndarray, src_mask: np.ndarray, dst_mask: np.ndarray, margin=1):
    """
    Combines the src_img into dst_img using the src_mask and dst_mask.
    """
    src_mask = src_mask > 0
    dst_mask = dst_mask > 0
    if margin > 0:
        dst_mask = nd.binary_dilation(dst_mask, iterations=margin)
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
    result = np.zeros(shape=(*dst_mask.shape, 3), dtype=src_img.dtype)
    result[rmin:rmax+1, cmin:cmax+1][dst_mask_clipped] = masked_src_img[dst_mask_clipped]
    return result

def im_fwd_gradient(image: np.ndarray):

    # CODE TO COMPLETE
    
    # Ux_i,j = Ui+1,j - Ui,j
    # we pad 1 at the end to keep same dimensions
    grad_i = np.pad(image[:, 1:], pad_width=((0, 0), (0, 1))) - np.pad(image[:, :-1], pad_width=((0, 0), (0, 1)))
    
    # Uy_i,j = Ui,j+1 - Ui,j
    # we pad 1 at the end to keep same dimensions
    grad_j = np.pad(image[1:, :], pad_width=((0, 1), (0, 0))) - np.pad(image[:-1, :], pad_width=((0, 1), (0, 0)))
    return grad_i, grad_j

def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray):

    # CODE TO COMPLETE
    # Vx_i,j - Vx_i-1,j
    # we pad 1 at the begining to keep same dimensions
    div_i = np.pad(im1[:, 1:], pad_width=((0, 0), (1, 0))) - np.pad(im1[:,:-1], pad_width=((0, 0), (1, 0)))
    # Vy_i,j - Vy_i-1,j
    # we pad 1 at the begining to keep same dimensions
    div_j = np.pad(im2[1:, :], pad_width=((1, 0), (0, 0))) - np.pad(im2[:-1,:], pad_width=((1, 0), (0, 0)))
    
    # plt.subplot(1,2, 1)
    # plt.imshow(div_i)

    # plt.subplot(1,2, 2)
    # plt.imshow(div_i)
    # plt.show()
    
    # plt.imshow(div_i + div_j)
    # plt.show()

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
    u1x, u1y = im_fwd_gradient(u1)
    u2x, u2y = im_fwd_gradient(u2)

    vi = np.zeros_like(u1x)
    vj = np.zeros_like(u1y)
    vi[mask] = u1x[mask]
    vi[~mask] = u2x[~mask]
    vj[mask] = u1y[mask]
    vj[~mask] = u2y[~mask]
    return vi, vj

def simple_poisson_solver(f_star: np.array, g: np.array, mask: np.array, mixed: bool=False):
    nj, ni = f_star.shape
    nPix = nj*ni
    A = sparse.lil_matrix((nPix, nPix), dtype=np.float64)
    b = np.zeros(shape=nPix, dtype=np.float64)
    
    inside_region = nd.binary_erosion(mask)
    inside_boundary = mask & ~inside_region
    outside_region = ~mask

    # CODE TO COMPLETE
    ## Boundary Condition
    for p in  np.argwhere(inside_boundary):
        ind = p[0] * ni + p[1]

        A[ind, ind] = 4
        v1 = g[*p] - g[p[0]+1, p[1]]
        v2 = g[*p] - g[p[0]-1, p[1]]
        v3 = g[*p] - g[p[0], p[1]+1]
        v4 = g[*p] - g[p[0], p[1]-1]
        b[ind] += v1 + v2 + v3 + v4

        neis = [(p[0]+1, p[1]), 
                (p[0]-1, p[1]),
                (p[0], p[1]+1),
                (p[0], p[1]-1)]
        
        for nei in neis:
            nei_ind = nei[0]*ni + nei[1]
            if mask[*nei]:
                A[ind, nei_ind] = -1
            else:
                b[ind] += f_star[*nei]
       

    ## Inside Condition
    for p in  np.argwhere(inside_region):
        ind = p[0] * ni + p[1]
        ind1 = (p[0]+1) * ni + p[1]
        ind2 = (p[0]-1) * ni + p[1]
        ind3 = (p[0]) * ni + p[1]+1
        ind4 = (p[0]) * ni + p[1]-1
        
        A[ind, ind] = 4
        A[ind, ind1] = -1
        A[ind, ind2] = -1
        A[ind, ind3] = -1
        A[ind, ind4] = -1

        v1 = g[*p] - g[p[0]+1, p[1]]
        v2 = g[*p] - g[p[0]-1, p[1]]
        v3 = g[*p] - g[p[0], p[1]+1]
        v4 = g[*p] - g[p[0], p[1]-1]
        b[ind] = v1 + v2 + v3 + v4


    ## Outside
    for p in  np.argwhere(outside_region):
        ind = p[0] * ni + p[1]
        A[ind, ind] = 1
        b[ind] = f_star[*p]
    
    A = A.tocsr()
    x = spsolve(A, b)
    result = np.reshape(x, f_star.shape)
    return result

def advanced_poisson_solver(f_star: np.array, g: np.array, mask: np.array, beta: float, mixed: bool=False):
    pass