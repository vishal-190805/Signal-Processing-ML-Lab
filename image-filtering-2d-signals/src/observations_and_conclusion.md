# Image Filtering and 2D Signal Analysis

This project implements various low-pass and high-pass filtering techniques to analyze 2D signals (images), focusing on noise reduction and edge detection.

## Objectives
The objective is to evaluate the effectiveness of different spatial filters in reducing Salt-and-Pepper noise and enhancing structural edges of an image.

## Analysis and Observations

### Noise Reduction (Low-Pass Filtering)
The experiment compared several filters to remove Salt-and-Pepper noise. The results are evaluated using **Peak Signal-to-Noise Ratio (PSNR)**.

| Filter | PSNR (dB) | Observation |
| :--- | :--- | :--- |
| **Noisy Image** | 17.16 | High presence of isolated bright/dark pixels. |
| **Mean Filter** | 24.92 | Reduces noise but causes significant blurring of edges. |
| **Gaussian Filter** | 24.53 | Smoother result, but less effective than Median for this noise type. |
| **Median (Inbuilt)** | **34.25** | **Best performance**; effectively removes outliers while preserving edges. |
| **Median (Custom)** | 33.54 | Close to inbuilt performance, demonstrating the neighborhood median logic. |

### Edge Detection (High-Pass Filtering)
Various operators were implemented to identify rapid intensity transitions:

*   **Laplacian**: Highlights rapid intensity changes; useful for overall edge detection.
*   **Sobel**: Directional filters (X and Y) that produce a gradient magnitude map.
*   **Prewitt**: Similar to Sobel but uses equal weights, providing a slightly different edge response.
*   **Canny**: Produces a clean, binary edge map by suppressing non-edge content.
*   **LoG (Laplacian of Gaussian)**: Combines Gaussian smoothing with Laplacian to reduce noise sensitivity during edge detection.
*   **Max Pooling**: Preserves the strongest local features, emphasizing bright structural elements.

## Conclusion
The results demonstrate that **Median Filtering** is the most effective approach for removing impulse (Salt-and-Pepper) noise. For edge detection, the **Canny** operator provides the most precise boundaries, while **Sobel** and **Prewitt** provide useful gradient information for directional analysis.
