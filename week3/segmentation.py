import numpy as np
import cv2

def initialize_level_set(shape, type='default'):
    X, Y = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]), indexing='xy')
    if type == 'default':
        phi = (-np.sqrt((X - np.round(shape[0] / 2)) ** 2 + (Y - np.round(shape[1] / 2)) ** 2) + 50)
    elif type == 'checkers':
        phi = np.sin(np.pi/5*Y) * np.sim(np.pi/5*X)
    else:
        phi = (-np.sqrt((X - np.round(shape[0] / 2)) ** 2 + (Y - np.round(shape[1] / 2)) ** 2) + 50)
    # Normalization of the initial phi to the range [-1, 1]
    min_val = np.min(phi)
    max_val = np.max(phi)
    phi = phi - min_val
    phi = 2 * phi / max_val
    phi = phi - 1
    return phi

def _dirac_delta(phi, epsilon):
    return (epsilon / (np.pi * (phi**2 + epsilon**2)))


def compute_binaryvalue_c1c2(f, phi, epsilon):
    H_phi_x = 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))

    # compute c1 and c2
    c1 = np.sum(H_phi_x * f) / (np.sum(H_phi_x) + 1e-10) #tiny number to avoid zero division
    c2 = np.sum((1 - H_phi_x) * f) / (np.sum(1 - H_phi_x) + 1e-10)
    return c1,c2

def _level_set_explicit_gradient(phi, f, c1, c2, mu, nu, l1, l2, epsilon):
    # Length Term Derivative
    phi_y, phi_x = np.gradient(phi)

    # compute gradient magnitude
    grad_phi_mag = np.sqrt(phi_x**2 + phi_y**2) + 1e-10  

    # normalized gradients
    norm_grad_x = phi_x / grad_phi_mag
    norm_grad_y = phi_y / grad_phi_mag

    # compute curvature
    div_x = np.gradient(norm_grad_x, axis=1)
    div_y = np.gradient(norm_grad_y, axis=0)
    div = div_x + div_y
    length_term = mu * div

    # Region Terms
    region1_term = -l1 * (f - c1)**2
    region2_term = l2 * (f - c2)**2

    # Area Term
    area_term = -nu

    derivative = _dirac_delta(phi, epsilon) * (length_term + area_term + region1_term + region2_term)
    #TODO set image boudnary
    return derivative

def explicit_gradient_descent_loop(f, phi, epsilon, mu, nu, lambda1, lambda2, dt, tol, iterMax):
    for iter in range(int(iterMax)):
        c1,c2 = compute_binaryvalue_c1c2(f= f, phi= phi, epsilon= epsilon)
        dphi_dt = _level_set_explicit_gradient(phi, f, c1, c2,mu,nu,lambda1,lambda2,epsilon)

        # update phi
        phi_old = phi.copy()
        phi = phi + dt * dphi_dt

        # converge
        if np.max(np.abs(phi - phi_old)) < tol:
            print('Converged at iteration', iter)
            break
        # display phi
        if iter % 1000 == 0:
            print('Iteration:', iter)
            phi_display = (phi - np.min(phi)) / (np.max(phi) - np.min(phi))
            phi_display = (phi_display * 255).astype(np.uint8)
            cv2.imshow('Phi', phi_display)
            cv2.waitKey(1)
    return phi

def get_level_set_averages(f, phi):
    c1 = np.mean(f[phi >= 0])
    c2 = np.mean(f[phi < 0])
    return (c1, c2)