# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:23:05 2026

@author: Shahjahan
"""

import numpy as np
import time
# from scipy.optimize import nnls
from scipy.special import ive
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt

# ============================================================
# Build T2 dictionary
# ============================================================
def build_t2_dictionary(Tdim, TE, nT2=150):
    t = np.arange(1, Tdim + 1)[:, None] * TE
    T2_values = np.logspace(np.log10(5e-3), np.log10(300e-3), nT2)
    A_raw = np.exp(-t / T2_values)
    col_norms = np.maximum(norm(A_raw, axis=0), 1e-12)
    A = np.nan_to_num(A_raw / col_norms)
    return A_raw, A, col_norms, T2_values

# ============================================================
# Gaussian ISTA
# ============================================================
def gaussian_ista_batch(Y_batch, A, AtA, tau, lam, max_iter, tol):

    n_voxels = Y_batch.shape[0]
    n_atoms = A.shape[1]

    # Initialize coefficients at zero
    F = np.zeros((n_voxels, n_atoms),
        dtype=np.float32)

    # This only depends on the current batch
    YAt = Y_batch @ A

    # Gaussian ISTA
    for iteration in range(max_iter):

        Grad = F @ AtA - YAt

        F_new = np.maximum(
            F - tau * (Grad + lam),
            0
        )

        rel = (
            np.linalg.norm(F_new - F, axis=1)
            /
            (np.linalg.norm(F, axis=1) + 1e-12)
        )

        F = F_new

        if np.all(rel < tol):
            print(
                f"Gaussian ISTA converged at "
                f"iteration {iteration + 1}"
            )
            break

    return F

# ============================================================
# Rician ISTA
# ============================================================
def rician_ista_batch(
    Y_batch,
    A,
    F_init,
    sigma_batch,
    lam,
    L_gauss,
    max_iter,
    tol
):

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    F = F_init.copy()

    # A.T is constant
    A_T = A.T

    # sigma^2 for each voxel
    s2 = np.maximum(
        sigma_batch[:, None] ** 2,
        1e-10
    )

    # Precompute inverse sigma^2
    inv_s2 = 1.0 / s2

    # Rician step size
    tau_rician = s2 / L_gauss

    # --------------------------------------------------
    # Quantities that remain constant during iterations
    # --------------------------------------------------
    Y_over_s2 = Y_batch * inv_s2

    # --------------------------------------------------
    # Rician ISTA
    # --------------------------------------------------

    for iteration in range(max_iter):

        # ----------------------------------------------
        # Reconstructed signal
        # ----------------------------------------------

        X = np.maximum( F @ A_T,
            1e-10 )

        # ----------------------------------------------
        # Bessel-function argument
        # ----------------------------------------------

        Z = np.clip(
            X * Y_over_s2,
            1e-10,
            700
        )

        # ----------------------------------------------
        # Rician Bessel ratio
        # ----------------------------------------------

        ratio = (
            ive(1, Z)
            /
            (ive(0, Z) + 1e-12)
        )


        signal_gradient = (
            X * inv_s2
            -
            Y_over_s2 * ratio
        )

        Grad = signal_gradient @ A

        # ----------------------------------------------
        # ISTA update
        # ----------------------------------------------

        F_new = np.maximum(
            F - tau_rician * (Grad + lam),
            0
        )

        # ----------------------------------------------
        # Convergence
        # ----------------------------------------------

        rel = (
            np.linalg.norm(
                F_new - F,
                axis=1
            )
            /
            (
                np.linalg.norm(F, axis=1)
                + 1e-12
            )
        )

        F = F_new

        if np.all(rel < tol):
            print(
                f"Rician ISTA converged at "
                f"iteration {iteration + 1}"
            )
            break

    else:
        print(
            f"Rician ISTA reached "
            f"max_iter = {max_iter}"
        )

    return F

# ============================================================
# Batch Rician-Lasso T2 denoising
# ============================================================
def rician_lasso_t2_denoise_batch(
    Y_batch,
    A,
    A_raw_T,
    AtA,
    inv_col_norms,
    lambda_gauss,
    lambda_rician,
    L_gauss,
    max_iter_gauss,
    max_iter_rician,
    tol_gauss,
    tol_rician
):

    # --------------------------------------------------
    # Normalize each voxel
    # --------------------------------------------------

    scales = np.maximum(
        np.max(Y_batch, axis=1, keepdims=True),
        1e-12
    )

    Y_batch_norm = Y_batch / scales

    # --------------------------------------------------
    # Estimate noise sigma
    # --------------------------------------------------

    sigma_batch = np.maximum(
        np.std(
            Y_batch_norm[:, -4:],
            axis=1
        ),
        1e-5
    )

    # --------------------------------------------------
    # Gaussian ISTA warm start
    # --------------------------------------------------

    tau_gauss = 1.0 / L_gauss

    F_init = gaussian_ista_batch(
        Y_batch_norm,
        A,
        AtA,
        tau_gauss,
        lambda_gauss,
        max_iter_gauss,
        tol_gauss
    )

    # --------------------------------------------------
    # Rician ISTA
    # --------------------------------------------------

    F = rician_ista_batch(
        Y_batch_norm,
        A,
        F_init,
        sigma_batch,
        lambda_rician,
        L_gauss,
        max_iter_rician,
        tol_rician
    )

    # --------------------------------------------------
    # Undo dictionary-column normalization
    # --------------------------------------------------

    F_orig = F * inv_col_norms

    # --------------------------------------------------
    # Reconstruct denoised signal
    # --------------------------------------------------

    X_denoised = (
        F_orig @ A_raw_T
    ) * scales

    return X_denoised, F_orig

# ============================================================
# Vectorized cube denoising with mask
# ============================================================
def denoise_cube_batch_vectorized(
    data_cube,
    TE,
    lambda_gauss=0.003,
    lambda_rician=0.005,
    max_iter_gauss=200,
    max_iter_rician=200,
    tol_gauss=1e-6,
    tol_rician=1e-6,
    batch_size=50000
):

    # ======================================================
    # 1. Dimensions
    # ======================================================

    Ydim, Xdim, Tdim, Zdim = data_cube.shape

    print(f"Data shape: {data_cube.shape}")

    # ======================================================
    # 2. Convert 4D cube -> voxels x echoes
    # ======================================================

    data_2d = (
        data_cube
        .transpose(0, 1, 3, 2)
        .reshape(-1, Tdim)
    )

    n_voxels = data_2d.shape[0]

    print(f"Total voxels: {n_voxels:,}")

    # ======================================================
    # 3. Build dictionary
    # ======================================================

    A_raw, A, col_norms, T2_vals = build_t2_dictionary(
        Tdim,
        TE
    )

    print(f"Dictionary shape: {A.shape}")

    # ======================================================
    # 4. Precompute dictionary
    # ======================================================

    # A.T @ A is constant for every batch
    AtA = A.T @ A

    # Transpose used during reconstruction
    A_raw_T = A_raw.T

    # Avoid division later
    inv_col_norms = 1.0 / col_norms

    # Gaussian Lipschitz constant
    L_gauss = norm(A, 2) ** 2

    print(f"Lipschitz constant: {L_gauss:.6e}")

    # print(f"Gaussian max iterations: {max_iter_gauss}")

    # print(f"Rician max iterations: {max_iter_rician}")

    # ======================================================
    # 5. Allocate output
    # ======================================================

    denoised_2d = np.zeros(
        data_2d.shape,
        dtype=np.float32
    )

    # ======================================================
    # 6. Process batches
    # ======================================================

    processed = 0

    total_batches = (
        n_voxels + batch_size - 1
    ) // batch_size

    for batch_number, start in enumerate(
        range(0, n_voxels, batch_size),
        start=1
    ):

        end = min(
            start + batch_size,
            n_voxels
        )

        print(f"\nBatch {batch_number}/{total_batches}: "
            f"voxels {start:,} -> {end:,}")

        # --------------------------------------------------
        # Get batch
        # --------------------------------------------------

        Y_batch = data_2d[start:end]

        # --------------------------------------------------
        # Denoise
        # --------------------------------------------------

        denoised_batch, _ = (
            rician_lasso_t2_denoise_batch(
                Y_batch,
                A,
                A_raw_T,
                AtA,
                inv_col_norms,
                lambda_gauss,
                lambda_rician,
                L_gauss,
                max_iter_gauss,
                max_iter_rician,
                tol_gauss,
                tol_rician
            ))

        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        denoised_2d[start:end] = denoised_batch

        processed = end

        print(f"Processed "
            f"{processed:,}/{n_voxels:,}")

    # ======================================================
    # 7. Restore original dimensions
    # ======================================================

    denoised_cube = (denoised_2d.reshape(Ydim,Xdim,Zdim,Tdim ).transpose(0,1,3,2))

    # ======================================================
    # 8. Safety clipping
    # ======================================================

    np.maximum(
        denoised_cube,
        0,
        out=denoised_cube
    )

    return denoised_cube

# ------------------------------------------------------------
# Plotting Function
# ------------------------------------------------------------
def plot_random_voxels(noisy, denoised, positions):
    

    # Echo times in ms
    t = (
        np.arange(1, data_cube.shape[2] + 1)
        * TE
        * 1000
    )

    plt.figure(figsize=(10, 6))

    for i, (y, x, z) in enumerate(positions):

        # Original noisy signal
        noisy_curve = data_cube[y, x, :, z]

        # Denoised signal
        denoised_curve = denoised[y, x, :, z]

        # Noisy
        plt.plot(
            t,
            noisy_curve,
            'o--',
            linewidth=1.5,
            markersize=5,
            label=f'Noisy {i+1}: ({y},{x},{z})'
        )

        # Denoised
        plt.plot(
            t,
            denoised_curve,
            '-',
            linewidth=2.5,
            label=f'Denoised {i+1}'
        )

    plt.xlabel(
        'Echo Time (ms)',
        fontsize=12
    )

    plt.ylabel(
        'Signal Intensity',
        fontsize=12
    )

    plt.title('Decay Curves at Selected Voxels',fontsize=14)

    plt.grid(True)

    plt.legend(fontsize=9)

    plt.tight_layout()

    plt.show()
    
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    start_time = time.perf_counter()

    # ============================================================
    # Load data
    # ============================================================

    data_path = (r'Spatially_Variable_Rician_Noise_SNR300_1000.npy')

    data_cube = np.load(data_path)

    print("Data shape:", data_cube.shape)
    print("Data dtype:", data_cube.dtype)

    # ============================================================
    # Parameters
    # ============================================================

    TE = 2.5e-3

    lambda_gauss = 0.03
    lambda_rician = 0.005

    # Start with these for the speed test
    max_iter_gauss = 100
    max_iter_rician = 100

    tol_gauss = 1e-6
    tol_rician = 1e-6

    batch_size = 100000

    # ============================================================
    # Denoising
    # ============================================================

    denoised = denoise_cube_batch_vectorized(
        data_cube=data_cube,
        TE=TE,
        lambda_gauss=lambda_gauss,
        lambda_rician=lambda_rician,
        max_iter_gauss=max_iter_gauss,
        max_iter_rician=max_iter_rician,
        tol_gauss=tol_gauss,
        tol_rician=tol_rician,
        batch_size=batch_size )
    
    positions = [
        (89, 70, 91),
        (50, 60, 80),
        (110, 120, 110),
        (60, 90, 50),
        (70, 80, 160)
    ]
    
    
    plot_random_voxels(data_cube, denoised, positions)
    # ============================================================
    # Save
    # ============================================================

    base_name = os.path.splitext(os.path.basename(data_path))[0]

    save_path = (
    Lasso_Denoiser{base_name}_Lasso_lam{lambda_rician}.npy')

    # Uncomment this line when you are happy with the result.
    np.save(save_path, denoised)

    print(f"\nOutput path:\n{save_path}")

    print(f"\nTotal runtime: "
        f"{time.perf_counter() - start_time:.2f} seconds")
