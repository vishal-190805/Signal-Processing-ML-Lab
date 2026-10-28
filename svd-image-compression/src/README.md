# SVD-Based Image Compression

This project implements an image compression algorithm using Singular Value Decomposition (SVD), demonstrating how dimensionality reduction can be used to represent images with minimal data loss.

## Objectives
The goal is to apply SVD to decompose an image matrix into its singular values and vectors, then reconstruct the image using only the top-k singular values to achieve compression.

## Methodology
The compression process follows these mathematical steps:

1.  **Image Representation**: Converting the image into a grayscale matrix $A$ of size $m \times n$.
2.  **SVD Decomposition**: Decomposing the matrix $A$ into three matrices: $A = U \Sigma V^T$
    *   $U$: Left singular vectors (orthogonal).
    *   $\Sigma$: Diagonal matrix containing singular values in descending order.
    *   $V^T$: Right singular vectors (orthogonal).
3.  **Low-Rank Approximation**: Selecting the top $k$ singular values from $\Sigma$ and discarding the rest.
4.  **Image Reconstruction**: Reconstructing the approximated image $A_k = U_k \Sigma_k V^T_k$.
5.  **Performance Evaluation**: Analyzing the trade-off between the compression ratio (number of singular values kept) and the resulting image quality.

## Key Findings
*   **Energy Concentration**: Most of the image's "energy" (information) is concentrated in the first few singular values.
*   **Compression Efficiency**: A significant reduction in data can be achieved while maintaining a visually acceptable representation of the original image.
*   **Visual Artifacts**: As $k$ decreases, the image becomes increasingly blurry, showing the loss of high-frequency structural details.

## Dependencies
*   `numpy`: Linear algebra operations and SVD computation.
*   `matplotlib`: Comparison of original vs. compressed images.
*   `PIL/OpenCV`: Image loading and processing.
