# results

## Documentation

EE69205 - Experiment 2
Name: Bagada Vishal Jayantilal
Roll No.: 23EE38004

OBSERVATIONS AND CONCLUSION
===========================

1. Salt-and-pepper noise:
   Salt-and-pepper noise introduces isolated bright and dark pixels.
   The noisy-image PSNR was 16.9797 dB.

2. Mean filter:
   The 3x3 mean filter averages neighboring pixels and reduces noise,
   but it also smooths image edges. The PSNR after filtering was
   25.1208 dB.

3. Gaussian filter:
   The specified 3x3 Gaussian kernel gives more weight to the center
   and nearby pixels than to farther pixels. The PSNR after filtering
   was 24.6741 dB.

4. Median filter:
   The median filter is particularly suitable for salt-and-pepper noise
   because isolated extreme values are rejected when the neighborhood
   median is selected.
   Inbuilt median PSNR  = 34.3821 dB.
   Custom median PSNR   = 34.0146 dB.

5. Best low-pass result by PSNR:
   Median Inbuilt filter, with PSNR = 34.3821 dB.

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
