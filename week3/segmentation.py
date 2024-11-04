import numpy as np

def get_level_set(shape, type='default'):
    Y, X = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]), indexing='ij')
    if type == 'default':
        return (-np.sqrt((Y - np.round(shape[0] / 2)) ** 2 + (X - np.round(shape[1] / 2)) ** 2) + 50)
    elif type == 'checkers':
        return np.sin(np.pi/5*Y) * np.sin(np.pi/5*X)
    return (-np.sqrt((Y - np.round(shape[0] / 2)) ** 2 + (X - np.round(shape[1] / 2)) ** 2) + 50)

def _dirac_function(phi, epsilon):
    return epsilon / (np.pi * (phi**2 + epsilon**2))


def _level_set_gradient(phi, f, c1, c2, mu, nu, l1, l2, epsilon=1.0):
    # Length Term Derivative
    grad_y, grad_x  = np.gradient(phi)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    norm_grad_x = grad_x / (magnitude + 1e-10)
    norm_grad_y = grad_y / (magnitude + 1e-10)
    div_y = np.gradient(norm_grad_y, axis=0)
    div_x = np.gradient(norm_grad_x, axis=1)
    div = div_x + div_y
    length_term = mu * div

    # Region Terms
    region1_term = -l1 * (f - c1)**2
    region2_term = l2 * (f - c2)**2

    # Area Term
    area_term = -nu

    derivative = _dirac_function(phi, epsilon) * (length_term + area_term + region1_term + region2_term)
    #TODO set image boudnary
    return derivative

def get_phi(phi,i,j):
    j_max, i_max = phi.shape
    if i < 0:
        i = 0
    elif i >= i_max:
        i = i_max-1
    
    if j < 0:
        j = 0
    elif j >= j_max:
        j = j_max-1
    
    return phi[j][i]


def get_A_value(phi_ij, phi_ip_j, phi_im_j, phi_i_jp, phi_i_jm, mu, n=1e-16):
    divisor = np.sqrt(np.float64(n + (phi_ip_j-phi_ij)**2 + ((phi_i_jp-phi_i_jm)/2)**2 ))
    return mu/divisor

def get_B_value(phi_ij, phi_ip_j, phi_im_j, phi_i_jp, phi_i_jm, mu, n=1e-16):
    divisor = np.sqrt(np.float64(n + ((phi_ip_j-phi_im_j)/2)**2 + (phi_ij-phi_ip_j)**2 ))
    return mu/divisor

def compute_new_phi_ij(phi, f, i, j, c1, c2, mu, nu, l1, l2, epsilon=1.0,dt=1e-2,region1_term=0,region2_term=0):

    dt_dirac_phi_ij = dt*_dirac_function(phi[i][j], epsilon)
    
    phi_ij = get_phi(phi,i,j)
    phi_ip_j = get_phi(phi,i+1,j)
    phi_im_j = get_phi(phi,i-1,j)
    phi_i_jp = get_phi(phi,i,j+1)
    phi_i_jm = get_phi(phi,i,j-1)

    Aij = get_A_value(phi_ij, phi_ip_j, phi_im_j, phi_i_jp, phi_i_jm, mu)
    A_left_n = get_A_value(phi_ij, phi_ip_j, phi_im_j, phi_i_jp, phi_i_jm, mu)
    Bij = get_B_value(phi_ij, phi_ip_j, phi_im_j, phi_i_jp, phi_i_jm, mu)
    B_up_n = get_B_value(phi_ij, phi_ip_j, phi_im_j, phi_i_jp, phi_i_jm, mu)

    factor_div = 1 + (dt_dirac_phi_ij*(Aij+A_left_n+Bij+B_up_n))
    AB_sum = Aij*phi_ip_j+A_left_n*phi_im_j+Bij*phi_i_jp+B_up_n*phi_i_jm


    factor_up = phi[i][j] + (dt_dirac_phi_ij * (AB_sum-nu+region1_term+region2_term))
    
    new_phi_ij = (factor_up)/(factor_div)
    return new_phi_ij


def _level_set_gauss_seidel(phi, f, c1, c2, mu, nu, l1, l2, epsilon=1.0, dt=1e-2, region1_term=None,region2_term=None):

    for i in range(len(phi)):
        for j in range(len(phi[0])):
            phi[i][j] = compute_new_phi_ij(phi, f, i, j, c1, c2, mu, nu, l1, l2, epsilon=1.0, dt=1e-2, region1_term=region1_term[i][j],region2_term=region2_term[i][j])


def compute_binaryvalue_c1c2(f, phi, epsilon):
    H_phi_x = 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))

    # compute c1 and c2
    c1 = np.sum(H_phi_x * f) / (np.sum(H_phi_x) + 1e-10) #tiny number to avoid zero division
    c2 = np.sum((1 - H_phi_x) * f) / (np.sum(1 - H_phi_x) + 1e-10)
    return c1,c2

def get_level_set_averages(f, phi):
    c1 = np.mean(f[phi >= 0])
    c2 = np.mean(f[phi < 0])
    return (c1, c2)