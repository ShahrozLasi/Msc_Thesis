# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:09:38 2026

@author: Shahjaha
"""

import numpy as np
import matplotlib.pyplot as plt
import time
#from scipy.special import ive
import os
# ==========================================================
# Important for local sigma
# ==========================================================
def estimate_sigma_from_curve(curve):

    tail = curve[-4:]

    med = np.median(tail)

    sigma = np.median(np.abs(tail - med)) / 0.6745

    return max(sigma, 1e-5)

# ==========================================================
# Curve normalization
# ==========================================================
def normalize_curve(curve, mode="max"):
    """
    Normalize a T2 decay curve.
    """

    if mode == "max":
        m = np.max(curve)
        if m > 0:
            return curve / m
        return curve

    elif mode == "none":
        return curve

    else:
        raise ValueError("Unknown normalization mode")


# ==========================================================
# 3D NLM using full T2 decay curves
# ==========================================================
def nlm_t2_curve_3d(
    data_cube,
    h=0.15,
    search_radius=2,
    normalize=True,
    norm_mode="max"
):

    Ydim, Xdim, Tdim, Zdim = data_cube.shape

    denoised_cube = np.zeros_like(
        data_cube,
        dtype=np.float32
    )


    # ======================================================
    # Normalize decay curves
    # ======================================================

    if normalize:

        if norm_mode == "max":

            max_vals = np.max(
                data_cube,
                axis=2,
                keepdims=True
            ).astype(np.float64)

            max_vals[max_vals <= 0] = 1.0


            data_norm = (
                data_cube.astype(np.float64)
                /
                max_vals
            )

        else:
            raise ValueError(
                "Only max normalization supported"
            )

    else:

        data_norm = data_cube.astype(np.float64)

        max_vals = np.ones(
            (Ydim,Xdim,1,Zdim),
            dtype=np.float64
        )


    total_voxels = (
        Ydim *
        Xdim *
        Zdim
    )

    print(
        f"Active voxels: {total_voxels:,}"
    )


    # ======================================================
    # Loop through voxels
    # ======================================================

    for idx in range(total_voxels):


        y = idx // (Xdim*Zdim)

        remainder = idx % (Xdim*Zdim)

        x = remainder // Zdim

        z = remainder % Zdim



        # --------------------------------------------------
        # Reference curve
        # --------------------------------------------------

        curve_i = data_norm[
            y,x,:,z
        ]


        # --------------------------------------------------
        # Search window
        # --------------------------------------------------

        y0 = max(
            0,
            y-search_radius
        )

        y1 = min(
            Ydim-1,
            y+search_radius
        )


        x0 = max(
            0,
            x-search_radius
        )

        x1 = min(
            Xdim-1,
            x+search_radius
        )


        z0 = max(
            0,
            z-search_radius
        )

        z1 = min(
            Zdim-1,
            z+search_radius
        )


        # --------------------------------------------------
        # Extract neighboring curves
        # --------------------------------------------------

        neighborhood = data_norm[
            y0:y1+1,
            x0:x1+1,
            :,
            z0:z1+1
        ]


        neighborhood = np.moveaxis(
            neighborhood,
            2,
            -1
        )


        neighborhood = neighborhood.reshape(
            -1,
            Tdim
        )


        # --------------------------------------------------
        # Coordinates of neighbors
        # --------------------------------------------------

        yy,xx,zz = np.meshgrid(
            np.arange(y0,y1+1),
            np.arange(x0,x1+1),
            np.arange(z0,z1+1),
            indexing="ij"
        )


        coords = np.column_stack(
            (
                yy.ravel(),
                xx.ravel(),
                zz.ravel()
            )
        )


        # remove reference voxel

        mask = ~(
            (coords[:,0]==y)
            &
            (coords[:,1]==x)
            &
            (coords[:,2]==z)
        )


        neighbors = neighborhood[
            mask
        ]


        if neighbors.shape[0] == 0:

            denoised_cube[
                y,x,:,z
            ] = data_cube[
                y,x,:,z
            ]

            continue



        # ==================================================
        # NLM similarity
        # ==================================================

        distance = np.mean(
            (
                neighbors -
                curve_i[None,:]
            )**2,
            axis=1
        )


        weights = np.exp(
            -distance /
            (2*h*h)
        )


        weights[
            ~np.isfinite(weights)
        ] = 0


        weight_sum = np.sum(
            weights
        )


        if weight_sum < 1e-12:

            weights = np.ones(
                neighbors.shape[0]
            )

            weight_sum = np.sum(
                weights
            )


        # ==================================================
        # Average normalized decay shape
        # ==================================================

        shape = np.sum(
            weights[:,None]
            *
            neighbors,
            axis=0
        )

        shape /= (
            weight_sum
            +
            1e-12
        )


        # ==================================================
        # Recover amplitude
        # ==================================================

        if normalize:

            amplitude_neighbors = max_vals[
                y0:y1+1,
                x0:x1+1,
                :,
                z0:z1+1
            ]

            amplitude_neighbors = np.moveaxis(
                amplitude_neighbors,
                2,
                -1
            )

            amplitude_neighbors = (
                amplitude_neighbors.reshape(-1)
            )


            amplitude_neighbors = (
                amplitude_neighbors[mask]
            )


            amplitude = np.median(
                amplitude_neighbors
            )


            result = (
                shape *
                amplitude
            )

        else:

            result = shape



        result[
            ~np.isfinite(result)
        ] = 0


        result = np.maximum(
            result,
            0
        )


        denoised_cube[
            y,x,:,z
        ] = result.astype(
            np.float32
        )



        if (idx+1)%100000==0:

            print(
                f"Processed "
                f"{idx+1:,}/{total_voxels:,}"
            )


    return denoised_cube
# ==========================================================
# Plot random voxel curves
# ==========================================================
def plot_voxel_curves(
    noisy,
    denoised,
    positions,
    h,
    search_radius
):

    Tdim = noisy.shape[2]

    time_axis = np.linspace(
        2,
        80,
        Tdim
    )


    fig, ax = plt.subplots(
        figsize=(10,7)
    )


    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple"
    ]


    for i, (y,x,z) in enumerate(positions):


        noisy_curve = noisy[
            y,x,:,z
        ].astype(float)


        denoised_curve = denoised[
            y,x,:,z
        ].astype(float)


        # normalize both using noisy maximum
        scale = np.max(noisy_curve)

        if scale > 0:

            noisy_curve /= scale
            denoised_curve /= scale



        ax.plot(
            time_axis,
            noisy_curve,
            "--",
            color=colors[i],
            alpha=0.5,
            label=f"Voxel {i+1} ({y},{x},{z}) noisy"
        )


        ax.plot(
            time_axis,
            denoised_curve,
            "-",
            color=colors[i],
            linewidth=2,
            label=f"Voxel {i+1} denoised"
        )



    ax.set_xlabel(
        "Echo time (ms)"
    )

    ax.set_ylabel(
        "Normalized signal"
    )

    ax.set_title(
        f"T2 decay curves\n"
        f"NLM h={h}, radius={search_radius}"
    )

    ax.grid(
        True,
        alpha=0.3
    )


    ax.legend(
        fontsize=8
    )


    plt.tight_layout()

    plt.show()
# ==========================================================
# RMSE calculation between clean and denoised data
# ==========================================================
def calculate_rmse(clean_cube, denoised_cube):

    error = denoised_cube - clean_cube

    mse = np.mean(error ** 2)

    rmse = np.sqrt(mse)

    return rmse

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    start_time = time.time()

    # ======================================================
    # Input file
    # ======================================================

    noisy_file_path = (r"F:\JIMM2\MWF_invivo\Python_V.1.7"
                       r"\synthetic data cube\Spatially_Variably_with_Rician_noise"
                       r"\Spatially_Variable_Rician_Noise_SNR100_1000.npy")

    # ======================================================
    # Load noisy data
    # ======================================================

    print("\n========================================")
    print("LOADING INPUT DATA")
    print("========================================")

    data_cube = np.load(
        noisy_file_path
    ).astype(np.float32)

    print(
        f"Noisy cube shape: {data_cube.shape}"
    )

    # ======================================================
    # NLM parameters
    # ======================================================

    h = 0.05

    search_radius = 2

    print("\n========================================")
    print("NLM PARAMETERS")
    print("========================================")

    print(
        f"h = {h}"
    )

    print(
        f"Search radius = {search_radius}")

    # ======================================================
    # Run NLM denoising
    # ======================================================

    print("\n========================================")
    print("STARTING NLM DENOISING")
    print("========================================")

    denoise_start = time.time()

    denoised = nlm_t2_curve_3d(
        data_cube=data_cube,
        h=h,
        search_radius=search_radius,
        normalize=True,
        norm_mode="max"
    )
    
    
    denoise_time = (
        time.time()
        - denoise_start
    )

    print("\n========================================")
    print("NLM DENOISING FINISHED")
    print("========================================")

    print(
        f"Denoising time: "
        f"{denoise_time:.2f} seconds"
    )

    # ======================================================
    # Echo-time axis
    # ======================================================

    Tdim = data_cube.shape[2]

    time_axis = np.linspace(
        2,
        80,
        Tdim
    )

    # ======================================================
    # Five voxel positions
    # ======================================================

    positions = [
        (89, 70, 91),
        (50, 60, 80),
        (110, 120, 110),
        (60, 90, 50),
        (70, 80, 160)
    ]
    
    plot_voxel_curves(
    noisy=data_cube,
    denoised=denoised,
    positions=positions,
    h=h,
    search_radius=search_radius
)
    # ======================================================
    # Runtime
    # ======================================================

    elapsed_time = (time.time() - start_time)

    print("\n========================================")
    print("FINISHED")
    print("========================================")

    print(f"Total elapsed time: "
        f"{elapsed_time:.2f} seconds")

    print(f"NLM denoising time: "
        f"{denoise_time:.2f} seconds" )
    
# ======================================================
# Save denoised dataset
# ======================================================

    # import os

    # Extract filename without extension
    base_name = os.path.splitext(
    os.path.basename(noisy_file_path))[0]


    # Folder where original data is stored
    output_folder = os.path.dirname(
    noisy_file_path)


    # New filename
    save_name = (f"{base_name}"
    f"_NLM_h{h:.2f}"
    f"_radius{search_radius}.npy")


    # Full output path
    save_path = os.path.join(
    output_folder,
    save_name)


    # Save denoised data
    np.save(
    save_path,
    denoised)


    print("\n========================================")
    print("DENOISED DATA SAVED")
    print("========================================")

    print(
    f"Output path:\n{save_path}"
)