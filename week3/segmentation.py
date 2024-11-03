import numpy as np

def get_level_set(shape, type='default'):
    Y, X = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]), indexing='ij')
    if type == 'default':
        return (-np.sqrt((Y - np.round(shape[0] / 2)) ** 2 + (X - np.round(shape[1] / 2)) ** 2) + 50)
    elif type == 'checkers':
        return np.sin(np.pi/5*Y) * np.sim(np.pi/5*X)
    return (-np.sqrt((Y - np.round(shape[0] / 2)) ** 2 + (X - np.round(shape[1] / 2)) ** 2) + 50)

def _dirac_function(phi, epsilon):
    return epsilon / (np.pi * (phi**2 + epsilon**2))


def _level_set_gradient(phi, f, c1, c2, mu, nu, l1, l2):
    # Length Term Derivative
    grad_x, grad_y = np.gradient(phi)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    norm_grad_x = grad_x / (magnitude + 1e-10)
    norm_grad_y = grad_y / (magnitude + 1e-10)
    div_x = np.gradient(norm_grad_x, axis=1)
    div_y = np.gradient(norm_grad_y, axis=0)
    div = div_x + div_y
    length_term = mu * div

    # Region Terms
    region1_term = -l1 * (f - c1)**2
    region2_term = l2 * (f - c2)**2

    # Area Term
    area_term = -nu

    derivative = _dirac_function(phi, 1) * (length_term + area_term + region1_term + region2_term)
    #TODO set image boudnary
    return derivative
    

def get_level_set_averages(f, phi):
    c1 = np.mean(f[phi >= 0])
    c2 = np.mean(f[phi < 0])
    return (c1, c2)