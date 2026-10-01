# -*- coding: utf-8 -*-
"""
Created on Mon 20 14:13:34 2026

@author: shahjaha
"""
import numpy as np
import time
import os
# from scipy.special import i0e, i1e
from Tikhonov_Denoising_Version_2_faster_batch_processing import rician_tikhonov_denoise_batch

# ============================================================
# Rician + Tikhonov batch denoiser
# ============================================================
# def rician_tikhonov_denoise_batch(Y_batch, lam, sigma_batch,max_iter=200, tau=0.0002):

#     eps = 1e-12

#     Y_batch = np.asarray(Y_batch, dtype=np.float32)
#     X = np.maximum(Y_batch.copy(), 1e-6)

#     sigma_batch = np.asarray(sigma_batch, dtype=np.float32).reshape(-1, 1)
#     sigma_batch = np.maximum(sigma_batch, 1e-6)

#     s2 = sigma_batch ** 2

#     for _ in range(max_iter):

#         # ----------------------------------------------------
#         # Rician likelihood gradient
#         # ----------------------------------------------------
#         z = (X * Y_batch) / (s2 + eps)
#         z = np.clip(z, 1e-10, 700)

#         I0 = i0e(z)
#         I1 = i1e(z)

#         ratio = I1 / (I0 + eps)

#         grad_data = (X / s2) - (Y_batch / s2) * ratio

#         # ----------------------------------------------------
#         # Tikhonov regularization gradient
#         # ----------------------------------------------------
#         grad_reg = np.zeros_like(X)

#         grad_reg[:, 1:-1] = 2*X[:,1:-1] - X[:,:-2] - X[:,2:]
#         grad_reg[:, 0]    = X[:,0] - X[:,1]
#         grad_reg[:, -1]   = X[:,-1] - X[:,-2]

#         # ----------------------------------------------------
#         # Normalize per voxel
#         # ----------------------------------------------------
#         grad_data /= (np.mean(np.abs(grad_data), axis=1, keepdims=True) + eps)
#         grad_reg  /= (np.mean(np.abs(grad_reg), axis=1, keepdims=True) + eps)

#         grad = grad_data + lam * grad_reg

#         grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)

#         X_new = X - tau * grad / (grad_norm + eps)
#         X_new = np.maximum(X_new, 0)

#         # convergence
#         rel = np.linalg.norm(X_new - X, axis=1) / (np.linalg.norm(X, axis=1) + eps)

#         X = X_new

#         if np.all(rel < 1e-6):
#             break

#     return X


# ============================================================
# Full cube denoising
# ============================================================
def denoise_cube_batch_2d(data_cube, lam,
                          max_iter=200,
                          tau=0.0002,
                          batch_size=50000):

    Ydim, Xdim, Tdim, Zdim = data_cube.shape

    print("Input shape:", data_cube.shape)

    # --------------------------------------------------------
    # Convert cube -> (N_voxels, T)
    # --------------------------------------------------------
    data_2d = data_cube.transpose(0,1,3,2).reshape(-1, Tdim)

    n_voxels = data_2d.shape[0]

    # --------------------------------------------------------
    # Estimate sigma per voxel from last 4 echoes
    # --------------------------------------------------------
    sigma = np.std(data_cube[:, :, -4:, :], axis=2)
    sigma = np.maximum(sigma, 1e-6)
    sigma_1d = sigma.reshape(-1)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    denoised_2d = np.zeros_like(data_2d, dtype=np.float32)

    print(f"Total voxels: {n_voxels:,}")

    # --------------------------------------------------------
    # Batch loop
    # --------------------------------------------------------
    for start in range(0, n_voxels, batch_size):

        end = min(start + batch_size, n_voxels)
        batch_idx = np.arange(start, end)

        Y_batch = data_2d[batch_idx]
        sigma_batch = sigma_1d[batch_idx]

        X_batch = rician_tikhonov_denoise_batch(
            Y_batch,
            lam=lam,
            sigma_batch=sigma_batch,
            max_iter=max_iter,
            tau=tau
        )

        denoised_2d[batch_idx] = X_batch

        print(f"Processed {end:,} / {n_voxels:,}")

    # --------------------------------------------------------
    # Back to cube shape
    # --------------------------------------------------------
    denoised_cube = (
        denoised_2d.reshape(Ydim, Xdim, Zdim, Tdim)
                   .transpose(0,1,3,2)
    )

    return denoised_cube


import matplotlib.pyplot as plt

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    start_time = time.time()

    # --------------------------------------------------------
    # Load cube
    # --------------------------------------------------------
    data_path = "F:\JIMM2\MWF_invivo\Python_V.1.7\synthetic data cube\Spatially_Variably_with_Rician_noise\Spatially_Variable_Rician_Noise_SNR100_1000.npy"
    data_cube = np.load(data_path)
    base_name = os.path.splitext(os.path.basename(data_path))[0]

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------
    lam = 10
    tau = 0.0002
    max_iter = 200
    batch_size = 100000

    # --------------------------------------------------------
    # Run denoising
    # --------------------------------------------------------
    denoised = denoise_cube_batch_2d(
        data_cube,
        lam=lam,
        tau=tau,
        max_iter=max_iter,
        batch_size=batch_size
    )

    # --------------------------------------------------------
    # Save denoised cube
    # --------------------------------------------------------
    save_path = rf"\\msg-filer6\scratch_360_days\JIMM2\MWF_invivo\Python_V.1.7\synthetic data cube\Tikhonov_Denoising/{base_name}_Tikhonov_lam{lam}_tau{tau}.npy"
    np.save(save_path, denoised)
    print("Saved:", save_path)
    

    # --------------------------------------------------------
    # Plot decay curves for selected positions
    # --------------------------------------------------------
    positions = [
        (89,70,91),   # Center
        (50,60,80),   # Front-left
        (110,120,110), # Right
        (60,90,50),    # Middle
        (70,80,160)    # Bottom
    ]

    plt.figure(figsize=(10,6))

    for pos in positions:
        x, y, z = pos
        raw_signal = data_cube[x, y, :, z]
        denoised_signal = denoised[x, y, :, z]

        plt.plot(raw_signal, label=f"Noisy {pos}")
        plt.plot(denoised_signal, '--', label=f"Denoised {pos}")

    plt.xlabel("Time Point")
    plt.ylabel("Normalized Signal")
    plt.title("Signal Decay Curves")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Runtime: {time.time() - start_time:.2f} sec")
    