"""
EE69205 - Signal Processing and System Design Laboratory
Experiment 2: Operations on 2D Signals - Image Filtering and Processing

Name    : Bagada Vishal Jayantilal
Roll No.: 23EE38004

Based on the provided IIT Kharagpur Experiment 2 lab manual.

Required packages:
    numpy
    matplotlib
    opencv-python
    scipy
    scikit-image

Place the grayscale sample image as "sample_image.tif" in the same folder
as this script, or change FILE_PATH below.
"""

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import gaussian_laplace, maximum_filter
from skimage.util import random_noise
from skimage.feature import canny


# ============================================================
# 0. CONFIGURATION
# ============================================================

NAME = "Bagada Vishal Jayantilal"
ROLL_NO = "23EE38004"

FILE_PATH = r"lena_color_512.tif"
OUTPUT_DIR = "experiment2_outputs"

# Salt-and-pepper probabilities specified in the manual
SALT_PROB = 0.02
PEPPER_PROB = 0.05

# Reproducible noise
RANDOM_SEED = 10

# Filters specified in the manual
MEAN_KERNEL = np.ones((3, 3), dtype=np.float64) / 9.0
GAUSSIAN_KERNEL = np.array(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]], dtype=np.float64
) / 16.0

LAPLACIAN_KERNEL = np.array(
    [[0, -1, 0],
     [-1, 4, -1],
     [0, -1, 0]], dtype=np.float64
)

SOBEL_X = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], dtype=np.float64
)

SOBEL_Y = np.array(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]], dtype=np.float64
)

PREWITT_X = np.array(
    [[-1, 0, 1],
     [-1, 0, 1],
     [-1, 0, 1]], dtype=np.float64
)

PREWITT_Y = np.array(
    [[-1, -1, -1],
     [0, 0, 0],
     [1, 1, 1]], dtype=np.float64
)


# ============================================================
# 1. UTILITY FUNCTIONS
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_image(filename, image, normalize=False):
    """Save an image safely as an 8-bit grayscale PNG."""
    arr = np.asarray(image)

    if normalize:
        arr = arr.astype(np.float64)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        else:
            arr = np.zeros_like(arr)
    else:
        arr = np.clip(arr, 0, 255)

    arr = arr.astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), arr)


def show_and_save(images, titles, filename, cmap="gray", normalize_flags=None):
    """
    Display and save multiple images in one row.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    if normalize_flags is None:
        normalize_flags = [False] * n

    for ax, image, title, norm in zip(
        axes, images, titles, normalize_flags
    ):
        display_image = np.asarray(image)

        if norm:
            display_image = display_image.astype(np.float64)
            mn, mx = display_image.min(), display_image.max()
            if mx > mn:
                display_image = (display_image - mn) / (mx - mn)
            else:
                display_image = np.zeros_like(display_image)

        ax.imshow(display_image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def psnr(original, processed, max_pixel=255.0):
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 10 log10(MAX^2 / MSE)
    """
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)

    mse = np.mean((original - processed) ** 2)

    if mse == 0:
        return float("inf")

    return 10.0 * np.log10((max_pixel ** 2) / mse)


def difference_map(original, processed):
    """Absolute pixel-wise difference."""
    return np.abs(
        np.asarray(original, dtype=np.float64)
        - np.asarray(processed, dtype=np.float64)
    )


def convolve2d_manual(img, kernel):
    """
    Basic 2D convolution from scratch using zero padding.

    The output has the same dimensions as the input.
    """
    image = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(
        image,
        ((ph, ph), (pw, pw)),
        mode="constant",
        constant_values=0
    )

    # True convolution: rotate kernel by 180 degrees.
    kernel_flipped = np.flipud(np.fliplr(kernel))

    output = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel_flipped)

    return output


def normalize_to_uint8(image):
    """Normalize arbitrary numeric image to [0,255]."""
    image = np.asarray(image, dtype=np.float64)
    mn, mx = image.min(), image.max()

    if mx <= mn:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - mn) / (mx - mn) * 255.0
    return normalized.astype(np.uint8)


# ============================================================
# 2. MEDIAN FILTER FROM SCRATCH
# ============================================================

def bubble_sort(values):
    """Bubble sort implementation used by the custom median filter."""
    arr = list(values)

    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        if not swapped:
            break

    return np.array(arr)


def median_bubble(neighborhood):
    """Calculate median after flattening and bubble sorting."""
    flat = np.asarray(neighborhood).flatten()
    sorted_flat = bubble_sort(flat)
    median_index = len(sorted_flat) // 2

    return sorted_flat[median_index]


def median_filter_custom(image, filter_size=3):
    """
    Median filter implemented from scratch.

    Zero padding is used at the borders.
    """
    image = np.asarray(image, dtype=np.float64)

    pad = filter_size // 2
    padded = np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0
    )

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded[
                i:i + filter_size,
                j:j + filter_size
            ]
            output[i, j] = median_bubble(neighborhood)

    return output.astype(np.uint8)


# ============================================================
# 3. MAX POOLING FROM SCRATCH
# ============================================================

def max_pooling_custom(image, pool_size=5, stride=1):
    """
    Max pooling implemented from scratch.

    This produces a downsampled output when stride > 1.
    The lab manual uses pool_size = 5.
    """
    image = np.asarray(image)

    h, w = image.shape

    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    if out_h <= 0 or out_w <= 0:
        raise ValueError("Pool size is larger than the image.")

    output = np.zeros((out_h, out_w), dtype=image.dtype)

    out_i = 0
    for i in range(0, h - pool_size + 1, stride):
        out_j = 0

        for j in range(0, w - pool_size + 1, stride):
            region = image[i:i + pool_size, j:j + pool_size]
            output[out_i, out_j] = np.max(region)
            out_j += 1

        out_i += 1

    return output


# ============================================================
# 4. LOAD IMAGE
# ============================================================

def load_grayscale_image():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(
            f"\nImage not found: {FILE_PATH}\n"
            "Download the grayscale sample image specified in the lab manual "
            "and place it at this path, or edit FILE_PATH at the top of this script."
        )

    img = cv2.imread(FILE_PATH, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("The image could not be read by OpenCV.")

    return img


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("EE69205 - EXPERIMENT 2")
    print("Operations on 2D Signals - Image Filtering and Processing")
    print("=" * 70)
    print(f"Name    : {NAME}")
    print(f"Roll No.: {ROLL_NO}")
    print()

    ensure_output_dir()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------
    img = load_grayscale_image()

    print(f"Image shape: {img.shape}")
    print(f"Image dtype : {img.dtype}")
    print()

    save_image("01_original.png", img)

    # --------------------------------------------------------
    # Add salt-and-pepper noise
    # --------------------------------------------------------
    np.random.seed(RANDOM_SEED)

    noisy_float = random_noise(
        img,
        mode="s&p",
        amount=SALT_PROB + PEPPER_PROB
    )

    noisy_img = (255 * noisy_float).astype(np.uint8)

    save_image("02_noisy.png", noisy_img)

    show_and_save(
        [img, noisy_img],
        ["Original Image", "Salt & Pepper Noisy Image"],
        "part1_original_noisy.png"
    )

    # --------------------------------------------------------
    # Part 1A: Mean filter
    # --------------------------------------------------------
    start = time.perf_counter()

    mean_filtered = convolve2d_manual(img=noisy_img, kernel=MEAN_KERNEL)

    mean_time = time.perf_counter() - start
    mean_filtered = np.clip(mean_filtered, 0, 255).astype(np.uint8)

    psnr_noisy = psnr(img, noisy_img)
    psnr_mean = psnr(img, mean_filtered)

    mean_diff_noisy = difference_map(img, noisy_img)
    mean_diff_filtered = difference_map(img, mean_filtered)

    save_image("03_mean_filtered.png", mean_filtered)
    save_image("04_mean_difference_noisy.png", mean_diff_noisy)
    save_image("05_mean_difference_filtered.png", mean_diff_filtered)

    show_and_save(
        [img, noisy_img, mean_filtered],
        ["Original", "Noisy", "Mean Filter"],
        "part1_mean_comparison.png"
    )

    show_and_save(
        [mean_diff_noisy, mean_diff_filtered],
        ["|Original - Noisy|", "|Original - Mean|"],
        "part1_mean_difference_maps.png",
        normalize_flags=[True, True]
    )

    # --------------------------------------------------------
    # Part 1B: Gaussian filter
    # --------------------------------------------------------
    start = time.perf_counter()

    gaussian_filtered = convolve2d_manual(
        img=noisy_img,
        kernel=GAUSSIAN_KERNEL
    )

    gaussian_time = time.perf_counter() - start
    gaussian_filtered = np.clip(
        gaussian_filtered, 0, 255
    ).astype(np.uint8)

    psnr_gaussian = psnr(img, gaussian_filtered)

    gaussian_diff_filtered = difference_map(
        img,
        gaussian_filtered
    )

    save_image("06_gaussian_filtered.png", gaussian_filtered)
    save_image(
        "07_gaussian_difference_filtered.png",
        gaussian_diff_filtered
    )

    show_and_save(
        [img, noisy_img, gaussian_filtered],
        ["Original", "Noisy", "Gaussian Filter"],
        "part1_gaussian_comparison.png"
    )

    show_and_save(
        [mean_diff_noisy, gaussian_diff_filtered],
        ["|Original - Noisy|", "|Original - Gaussian|"],
        "part1_gaussian_difference_maps.png",
        normalize_flags=[True, True]
    )

    # --------------------------------------------------------
    # Part 1C: Median filter - inbuilt
    # --------------------------------------------------------
    from scipy.ndimage import median_filter

    start = time.perf_counter()

    median_builtin = median_filter(noisy_img, size=3)

    median_builtin_time = time.perf_counter() - start

    median_builtin = median_builtin.astype(np.uint8)
    psnr_median_builtin = psnr(img, median_builtin)

    median_builtin_diff = difference_map(
        img,
        median_builtin
    )

    save_image("08_median_builtin.png", median_builtin)
    save_image(
        "09_median_builtin_difference.png",
        median_builtin_diff
    )

    # --------------------------------------------------------
    # Part 1D: Median filter - custom bubble sort
    # --------------------------------------------------------
    start = time.perf_counter()

    median_custom = median_filter_custom(
        noisy_img,
        filter_size=3
    )

    median_custom_time = time.perf_counter() - start

    psnr_median_custom = psnr(img, median_custom)

    median_custom_diff = difference_map(
        img,
        median_custom
    )

    save_image("10_median_custom.png", median_custom)
    save_image(
        "11_median_custom_difference.png",
        median_custom_diff
    )

    show_and_save(
        [
            img,
            noisy_img,
            median_builtin,
            median_custom
        ],
        [
            "Original",
            "Noisy",
            "Median - Inbuilt",
            "Median - Custom"
        ],
        "part1_median_comparison.png"
    )

    show_and_save(
        [
            median_builtin_diff,
            median_custom_diff
        ],
        [
            "|Original - Median Inbuilt|",
            "|Original - Median Custom|"
        ],
        "part1_median_difference_maps.png",
        normalize_flags=[True, True]
    )

    # --------------------------------------------------------
    # Part 1 PSNR results
    # --------------------------------------------------------
    print("-" * 70)
    print("PART 1 - LOW-PASS FILTERS")
    print("-" * 70)
    print(f"PSNR of noisy image          : {psnr_noisy:.4f} dB")
    print(f"PSNR after mean filter       : {psnr_mean:.4f} dB")
    print(f"PSNR after Gaussian filter   : {psnr_gaussian:.4f} dB")
    print(f"PSNR after median (inbuilt)  : {psnr_median_builtin:.4f} dB")
    print(f"PSNR after median (custom)   : {psnr_median_custom:.4f} dB")
    print()

    # --------------------------------------------------------
    # Part 2A: Laplacian filter
    # --------------------------------------------------------
    laplacian_response = convolve2d_manual(
        noisy_img,
        LAPLACIAN_KERNEL
    )

    laplacian_display = normalize_to_uint8(
        np.abs(laplacian_response)
    )

    psnr_laplacian = psnr(img, laplacian_display)

    save_image(
        "12_laplacian_response.png",
        laplacian_display
    )

    show_and_save(
        [img, noisy_img, laplacian_display],
        ["Original", "Noisy", "Laplacian"],
        "part2_laplacian.png"
    )

    # --------------------------------------------------------
    # Part 2B: Sobel filter
    # --------------------------------------------------------
    sobel_x = convolve2d_manual(img, SOBEL_X)
    sobel_y = convolve2d_manual(img, SOBEL_Y)

    sobel_magnitude = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    sobel_x_display = normalize_to_uint8(np.abs(sobel_x))
    sobel_y_display = normalize_to_uint8(np.abs(sobel_y))
    sobel_magnitude_display = normalize_to_uint8(sobel_magnitude)

    save_image("13_sobel_x.png", sobel_x_display)
    save_image("14_sobel_y.png", sobel_y_display)
    save_image(
        "15_sobel_magnitude.png",
        sobel_magnitude_display
    )

    show_and_save(
        [
            img,
            sobel_x_display,
            sobel_y_display,
            sobel_magnitude_display
        ],
        [
            "Original",
            "Sobel X",
            "Sobel Y",
            "Sobel Gradient Magnitude"
        ],
        "part2_sobel.png"
    )

    # --------------------------------------------------------
    # Part 2C: Prewitt filter
    # --------------------------------------------------------
    prewitt_x = convolve2d_manual(img, PREWITT_X)
    prewitt_y = convolve2d_manual(img, PREWITT_Y)

    prewitt_magnitude = np.sqrt(
        prewitt_x ** 2 + prewitt_y ** 2
    )

    prewitt_x_display = normalize_to_uint8(np.abs(prewitt_x))
    prewitt_y_display = normalize_to_uint8(np.abs(prewitt_y))
    prewitt_magnitude_display = normalize_to_uint8(
        prewitt_magnitude
    )

    save_image("16_prewitt_x.png", prewitt_x_display)
    save_image("17_prewitt_y.png", prewitt_y_display)
    save_image(
        "18_prewitt_magnitude.png",
        prewitt_magnitude_display
    )

    show_and_save(
        [
            img,
            prewitt_x_display,
            prewitt_y_display,
            prewitt_magnitude_display
        ],
        [
            "Original",
            "Prewitt X",
            "Prewitt Y",
            "Prewitt Gradient Magnitude"
        ],
        "part2_prewitt.png"
    )

    # --------------------------------------------------------
    # Part 3A: Canny edge detector
    # --------------------------------------------------------
    # The lab manual imports canny from skimage.feature.
    # skimage's Canny uses sigma rather than OpenCV's threshold API.
    sigma = 1

    canny_edges = canny(
        img.astype(np.float64) / 255.0,
        sigma=sigma
    )

    canny_display = (canny_edges * 255).astype(np.uint8)

    save_image("19_canny_edges.png", canny_display)

    show_and_save(
        [img, canny_display],
        ["Original", "Canny Edge Detected"],
        "part3_canny.png"
    )

    # --------------------------------------------------------
    # Part 3B: Laplacian of Gaussian (LoG)
    # --------------------------------------------------------
    sigma_log = 1

    log_filtered_img = gaussian_laplace(
        img.astype(np.float64),
        sigma=sigma_log
    )

    log_display = normalize_to_uint8(
        np.abs(log_filtered_img)
    )

    save_image(
        "20_log_filtered.png",
        log_display
    )

    show_and_save(
        [img, log_display],
        ["Original", "LoG Filtered"],
        "part3_log.png"
    )

    # --------------------------------------------------------
    # Part 3C: Max pooling
    # --------------------------------------------------------
    pool_size = 5

    # Manual says maximum_filter(img, size=pool_size).
    # This is a local max filter and retains the same dimensions.
    max_pooled_builtin = maximum_filter(
        img,
        size=pool_size
    ).astype(np.uint8)

    # Custom implementation with stride=1 gives the same conceptual
    # local-max operation but removes the border where a full window
    # cannot fit.
    start = time.perf_counter()

    max_pooled_custom = max_pooling_custom(
        img,
        pool_size=pool_size,
        stride=1
    )

    max_pool_custom_time = time.perf_counter() - start

    save_image(
        "21_max_pool_builtin.png",
        max_pooled_builtin
    )

    save_image(
        "22_max_pool_custom.png",
        max_pooled_custom
    )

    show_and_save(
        [img, max_pooled_builtin],
        ["Original", "Max Pooling - Inbuilt"],
        "part3_maxpool_builtin.png"
    )

    show_and_save(
        [img, max_pooled_custom],
        ["Original", "Max Pooling - Custom (Stride 1)"],
        "part3_maxpool_custom.png"
    )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------
    results = [
        ["Metric", "Value"],
        ["Name", NAME],
        ["Roll No", ROLL_NO],
        ["Image height", img.shape[0]],
        ["Image width", img.shape[1]],
        ["Salt probability", SALT_PROB],
        ["Pepper probability", PEPPER_PROB],
        ["Noisy PSNR (dB)", psnr_noisy],
        ["Mean filter PSNR (dB)", psnr_mean],
        ["Gaussian filter PSNR (dB)", psnr_gaussian],
        ["Median inbuilt PSNR (dB)", psnr_median_builtin],
        ["Median custom PSNR (dB)", psnr_median_custom],
        ["Laplacian display PSNR (dB)", psnr_laplacian],
        ["Mean custom filter time (s)", mean_time],
        ["Gaussian custom filter time (s)", gaussian_time],
        ["Median inbuilt time (s)", median_builtin_time],
        ["Median custom time (s)", median_custom_time],
        ["Max pooling custom time (s)", max_pool_custom_time],
    ]

    with open(
        os.path.join(OUTPUT_DIR, "results.csv"),
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerows(results)

    # --------------------------------------------------------
    # Automatic observation report
    # --------------------------------------------------------
    best_filter = max(
        [
            ("Mean", psnr_mean),
            ("Gaussian", psnr_gaussian),
            ("Median Inbuilt", psnr_median_builtin),
            ("Median Custom", psnr_median_custom),
        ],
        key=lambda x: x[1]
    )

    observations = f"""
EE69205 - Experiment 2
Name: {NAME}
Roll No.: {ROLL_NO}

OBSERVATIONS AND CONCLUSION
===========================

1. Salt-and-pepper noise:
   Salt-and-pepper noise introduces isolated bright and dark pixels.
   The noisy-image PSNR was {psnr_noisy:.4f} dB.

2. Mean filter:
   The 3x3 mean filter averages neighboring pixels and reduces noise,
   but it also smooths image edges. The PSNR after filtering was
   {psnr_mean:.4f} dB.

3. Gaussian filter:
   The specified 3x3 Gaussian kernel gives more weight to the center
   and nearby pixels than to farther pixels. The PSNR after filtering
   was {psnr_gaussian:.4f} dB.

4. Median filter:
   The median filter is particularly suitable for salt-and-pepper noise
   because isolated extreme values are rejected when the neighborhood
   median is selected.
   Inbuilt median PSNR  = {psnr_median_builtin:.4f} dB.
   Custom median PSNR   = {psnr_median_custom:.4f} dB.

5. Best low-pass result by PSNR:
   {best_filter[0]} filter, with PSNR = {best_filter[1]:.4f} dB.

6. Laplacian:
   The Laplacian is a second-order high-pass operator and emphasizes
   rapid intensity changes and edges. It is not primarily a denoising
   filter, so its PSNR relative to the original image should not be
   interpreted in the same way as the low-pass denoising results.

7. Sobel:
   Sobel-X emphasizes vertical intensity transitions and Sobel-Y
   emphasizes horizontal intensity transitions. Their combination
   gives the gradient magnitude.

8. Prewitt:
   Prewitt-X and Prewitt-Y similarly detect directional intensity
   transitions. Compared with Sobel, the kernels use equal weights
   within each directional row/column, so Sobel generally gives more
   emphasis to the center row/column.

9. Canny:
   Canny produces a binary edge map and highlights significant
   boundaries while suppressing much of the non-edge content.

10. LoG:
    Laplacian of Gaussian first smooths the image with a Gaussian
    conceptually and then applies the Laplacian. It highlights rapid
    intensity changes while reducing sensitivity to very small-scale
    noise compared with a raw Laplacian.

11. Max pooling:
    A maximum operation over a local 5x5 region preserves the largest
    intensity value in each neighborhood. It therefore emphasizes
    strong local bright features. The provided manual's inbuilt
    maximum_filter operation is a local maximum filter; it does not
    reduce image dimensions unless an additional stride/slicing step
    is applied.

12. Inbuilt vs custom implementations:
    The inbuilt functions are generally faster because they are highly
    optimized. The custom implementations demonstrate the underlying
    pixel-neighborhood operations and are useful for understanding
    the algorithms.

Conclusion:
    The experiment demonstrates low-pass filtering for noise reduction,
    high-pass filtering for edge enhancement, Canny and LoG for edge
    detection, and max filtering/pooling for retaining strong local
    features.
"""

    with open(
        os.path.join(OUTPUT_DIR, "observations_and_conclusion.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(observations.strip())

    print(observations)

    print("=" * 70)
    print(f"All output files have been saved in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()