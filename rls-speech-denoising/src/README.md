# RLS Adaptive Filter for Speech Denoising

This project implements a Recursive Least Squares (RLS) adaptive filter to remove additive noise from speech signals in real-time.

## Objectives
The primary goal is to implement an adaptive filter that can track non-stationary noise and subtract it from a primary signal to recover clean speech.

## Methodology
The RLS algorithm is implemented as follows:

1.  **Signal Model**: Using a primary signal (speech + noise) and a reference signal (noise only).
2.  **RLS Algorithm**:
    *   **Initialization**: Setting the initial weights and the inverse correlation matrix $P(0) = \delta^{-1} I$.
    *   **Weight Update**: Recursively updating the filter coefficients based on the error signal $e(n)$ and the gain vector $k(n)$.
    *   **Forgetting Factor**: Implementing a forgetting factor $\lambda$ to control how quickly the filter forgets old data, allowing it to adapt to changing noise environments.
3.  **Performance Metrics**: Evaluating the denoising quality using the Signal-to-Noise Ratio (SNR) improvement.
4.  **Spectral Comparison**: Plotting the power spectrum of the noise-corrupted vs. the denoised speech.

## Key Results
The RLS filter demonstrates significantly faster convergence compared to the Least Mean Squares (LMS) algorithm, providing a cleaner output with minimal lag, especially in environments with dynamic noise profiles.

## Dependencies
*   `numpy`: Matrix operations and RLS weight updates.
*   `scipy.io.wavfile`: Reading and writing `.wav` audio files.
*   `matplotlib`: Visualization of time-domain and frequency-domain results.
