# Bearing Fault Diagnosis using Signal Processing

This implementation focuses on detecting bearing faults (inner and outer raceway) and comparing them against baseline healthy conditions using signal processing techniques.

## Objectives
The primary goal is to implement a complete signal processing pipeline to identify bearing faults from raw vibration data, as specified in the EE69205 SPSD Lab manual.

## Methodology
The following pipeline is implemented for each condition (Healthy, Inner Race Fault, Outer Race Fault):

1.  **Signal Acquisition**: Loading `.mat` files from the MFPT bearing dataset.
2.  **Domain Analysis**: 
    *   **Time Domain**: Plotting raw signals to observe impact patterns.
    *   **Frequency Domain**: Implementing a **recursive radix-2 Cooley-Tukey FFT** from scratch to analyze the spectrum.
    *   **Time-Frequency Analysis**: Implementing a **Short-Time Fourier Transform (STFT)** from scratch to observe frequency shifts over time.
3.  **Resonance Filtering**: Designing a bandpass FIR filter around the resonance band to isolate fault-induced impacts.
4.  **Envelope Analysis**: 
    *   Implementing the **Hilbert Transform** from scratch.
    *   Computing the envelope of the filtered signal to extract the repetition frequency of the faults.
5.  **Fault Verification**: Computing the power spectrum of the envelope and overlaying theoretical harmonics (BPFI / BPFO) for verification.
6.  **Statistical Analysis**: Computing the **Kurtosis** of signals to quantify the "peakiness" associated with faults.

## Technical Implementation
To adhere to laboratory requirements, the following components were implemented **from scratch** (without relying on `numpy.fft` or `scipy.fft`):
*   Fast Fourier Transform (FFT)
*   Short-Time Fourier Transform (STFT)
*   Hilbert Transform
*   Power Spectrum Calculation

## Dependencies
*   `numpy`: Array manipulations.
*   `scipy`: Used only for `.mat` file loading, FIR filter design (`firwin`), and `filtfilt`.
*   `matplotlib`: Data visualization.
