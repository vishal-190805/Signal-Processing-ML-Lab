# Spectral Density Analysis

This project implements techniques for analyzing the power spectral density (PSD) of signals to identify dominant frequency components and noise characteristics.

## Objectives
The objective is to analyze the frequency content of signals using the Periodogram and Welch's method to differentiate between signal and noise in a stochastic environment.

## Methodology
The analysis pipeline implements the following:

1.  **Signal Generation/Loading**: Processing time-domain signals.
2.  **Windowing**: Applying window functions (e.g., Hamming or Hann) to reduce spectral leakage at the edges of the signal segments.
3.  **Spectral Estimation**:
    *   **Periodogram**: Computing the basic power spectrum using the magnitude squared of the DFT.
    *   **Welch's Method**: Improving the estimate by averaging periodograms of overlapping segments to reduce variance.
4.  **Peak Detection**: Identifying the dominant frequencies that characterize the signal source.
5.  **SNR Calculation**: Estimating the Signal-to-Noise Ratio based on the power of the signal peak versus the noise floor.

## Technical Implementation
The implementation focuses on the trade-off between frequency resolution and variance reduction, demonstrating how averaging segments in Welch's method produces a smoother, more reliable spectrum than a single periodogram.

## Dependencies
*   `numpy`: Numerical operations and DFT calculations.
*   `matplotlib`: Plotting the Power Spectral Density (PSD).
*   `scipy.signal`: Utilizing `welch` for spectral estimation.
