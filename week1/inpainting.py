from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import cv2

@dataclass
class Parameters:
    hi: float
    hj: float

def laplace_equation(f, mask, param):
    ni = f.shape[0]
    nj = f.shape[1]

    # Add ghost boundaries on the image (for the boundary conditions)
    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0]
    nj_ext = f_ext.shape[1]
    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f

    # Add ghost boundaries on the mask
    mask_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = mask_ext.shape[0]
    ndj_ext = mask_ext.shape[1]
    print(ndi_ext, ndj_ext)
    mask_ext[1 : ndi_ext - 1, 1 : ndj_ext - 1] = mask

    # Store memory for the A matrix and the b vector
    nPixels = (ni+2)*(nj+2) # Number of pixels

    # We will create A sparse, this is the number of nonzero positions
    # idx_Ai: Vector for the nonZero i index of matrix A
    # idx_Aj: Vector for the nonZero j index of matrix A
    # a_ij: Vector for the value at position ij of matrix A

    b = np.zeros(nPixels, dtype=float)

    # Vector counter
    north_idx_Ai=[]
    north_idx_Aj=[]
    north_a_ij=[]

    # North side boundary conditions
    for j in range(nj_ext):
        p = j * ni_ext
        north_idx_Ai.append(p)
        north_idx_Aj.append(p)
        north_a_ij.append(1)

        north_idx_Ai.append(p)
        north_idx_Aj.append(p+1)
        north_a_ij.append(-1)

    print("North:")
    print(list(zip(north_idx_Ai, north_idx_Aj, north_a_ij))[:10])
    print()
    south_idx_Ai=[]
    south_idx_Aj=[]
    south_a_ij=[]
    
    # South side boundary conditions
    for j in range(nj_ext):
        p = j * (ni_ext) + (ni_ext-1)
        south_idx_Ai.append(p)
        south_idx_Aj.append(p)
        south_a_ij.append(1)
        
        north_idx_Ai.append(p)
        north_idx_Aj.append(p-1)
        north_a_ij.append(-1)
    
    print("South:")
    print(list(zip(south_idx_Ai, south_idx_Aj, south_a_ij))[:10])
    print()

    west_idx_Ai=[]
    west_idx_Aj=[]
    west_a_ij=[]
    # West side boundary conditions
    for i in range(1, ni_ext-1):
        p = i
        west_idx_Ai.append(p)
        west_idx_Aj.append(p)
        west_a_ij.append(1)

        west_idx_Ai.append(p)
        west_idx_Aj.append(p+ndi_ext)
        west_a_ij.append(-1)

    print("West:")
    print(list(zip(south_idx_Ai, south_idx_Aj, south_a_ij))[:10])
    print()

    east_idx_Ai=[]
    east_idx_Aj=[]
    east_a_ij=[]
    # East side boundary conditions
    for i in range(1, ni_ext-1):
        p = (nj_ext - 1) * (ni_ext) + i
        east_idx_Ai.append(p)
        east_idx_Aj.append(p)
        east_a_ij.append(1)

        west_idx_Ai.append(p)
        west_idx_Aj.append(p-ndi_ext)
        west_a_ij.append(-1)

    print("East:")
    print(list(zip(south_idx_Ai, south_idx_Aj, south_a_ij))[:10])
    print()

    idx_Ai = north_idx_Ai + south_idx_Ai + west_idx_Ai + east_idx_Ai
    idx_Aj = north_idx_Aj + south_idx_Aj + west_idx_Aj + east_idx_Aj
    a_ij = north_a_ij + south_a_ij + west_a_ij + east_a_ij
    # Looping over the pixels
    for j in range(1, nj_ext - 1):
        for i in range(1, ni_ext - 1):

            # from image matrix (i, j) coordinates to vectorial(p) coordinate
            p = j * ni_ext + i

            if mask_ext[i, j] == 1: # we have to in-paint this pixel
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
                # COMPLETE THE CODE

                #-4*U(i,j) + U(i+1,j) + U(i-1,j) + U(i,j+1) + U(i,j-1) = 0
                idx_Ai.append(p)
                idx_Aj.append(p) 
                a_ij.append(-4)

                idx_Ai.append(p)
                idx_Aj.append(p - 1) 
                a_ij.append(1)

                idx_Ai.append(p)
                idx_Aj.append(p + 1) 
                a_ij.append(1)

                idx_Ai.append(p)
                idx_Aj.append(p - ni_ext)
                a_ij.append(1)

                idx_Ai.append(p)
                idx_Aj.append(p + ni_ext) 
                a_ij.append(1)
                b[p] = 0

            else: # we do not have to in-paint this pixel
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
                # COMPLETE THE CODE
                
                #U(i,j) = F(i,j)
                idx_Ai.append(p)
                idx_Aj.append(p)
                a_ij.append(1)
                
                b[p] = f_ext[i, j]

    # COMPLETE THE CODE (fill out the interrogation marks ???)
    print(nPixels)
    print(max(idx_Ai), min(idx_Ai))
    print(max(idx_Aj), min(idx_Aj))
    A = sparse(idx_Ai, idx_Aj, a_ij, nPixels, nPixels)
    x = spsolve(A, b)
    u_ext = np.reshape(x,(ni+2, nj+2), order='F')
    u_ext_i = u_ext.shape[0]
    u_ext_j = u_ext.shape[1]
    u = np.full((ni, nj), u_ext[1:u_ext_i-1, 1:u_ext_j-1], order='F')
    return u

def sparse(i, j, v, m, n):
    """
    Create and compress a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values
            Size n1
        j: 1-D array representing the index 2 values
            Size n1
        v: 1-D array representing the values
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return csr_matrix((v, (i, j)), shape=(m, n))