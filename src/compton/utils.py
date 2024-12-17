import numpy as np

def Stokes_parameters(A):
    A_filter = np.zeros_like(A)
    A_filter[0] = (A[0] - 1j*A[1]) / np.sqrt(2)
    A_filter[1] = (A[0] + 1j*A[1]) / np.sqrt(2)
    A_filter[2] = A[2]
    del A
                
    return A_filter

def Lorentz_harm(A, w_grid, theta_grid, phi_grid, gamma=1, forward=False):
    n_theta, n_phi = A.shape[1:]
    if forward:
        forward = 1
    else:
        forward = -1
      
    beta = np.sqrt(1. - 1. / gamma**2)
    A_L = np.zeros_like(A)
    w_L, theta_L = np.zeros_like(w_grid), np.zeros_like(theta_grid)
    
    cos_theta_L = (np.cos(theta_grid) + forward*beta) / (1 + forward*beta*np.cos(theta_grid))
    theta_L = np.arccos(cos_theta_L)
    for i in range(theta_grid.shape[0]):
        angle = 1 + forward*beta*np.cos(theta_grid[i])
        w_L[i] = w_grid[i] * gamma * angle
        A_L[3:,i] = A[3:,i] * gamma * angle
    
    phi = phi_grid.reshape((1,n_phi))
    theta_L_ = theta_L.reshape((n_theta,1))
    A_L[0] = A_L[3] * np.cos(theta_L_) * np.cos(phi) - A_L[4] * np.sin(phi)
    A_L[1] = A_L[3] * np.cos(theta_L_) * np.sin(phi) + A_L[4] * np.cos(phi)
    A_L[2] = -A_L[3] * np.sin(theta_L_)

    return A_L, w_L, theta_L