import numpy as np
import matplotlib.pyplot as plt

def compute_dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    kernel = np.exp(-1j * 2 * np.pi * k * n / N)
    return np.dot(kernel, x)

def compute_idft(X):
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))

    kernel = np.exp(1j * 2 * np.pi * k * n / N)
    return np.real(np.dot(kernel, X)) / N

def compute_dct2(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    w = np.sqrt(2.0 / N) * np.ones((N, 1))
    w[0] = 1.0 / np.sqrt(N)

    kernel = w * np.cos((np.pi * k * (2 * n + 1)) / (2.0 * N))
    return np.dot(kernel, x)

def compute_idct2(X):
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))

    w = np.sqrt(2.0 / N) * np.ones((N, 1))
    w[0] = 1.0 / np.sqrt(N)

    kernel = w * np.cos((np.pi * k * (2 * n + 1)) / (2.0 * N))
    return np.dot(kernel.T, X)

def compute_dst2(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    w = np.sqrt(2.0 / N) * np.ones((N, 1))
    w[-1] = 1.0 / np.sqrt(N)

    kernel = w * np.sin((np.pi * (k + 1) * (2 * n + 1)) / (2.0 * N))
    return np.dot(kernel, x)

def compute_idst2(X):
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))

    w = np.sqrt(2.0 / N) * np.ones((N, 1))
    w[-1] = 1.0 / np.sqrt(N)

    kernel = w * np.sin((np.pi * (k + 1) * (2 * n + 1)) / (2.0 * N))
    return np.dot(kernel.T, X)

F0 = 1.0
Omega = 2 * np.pi * F0
f_max = 3 * F0
Fs = 20 * f_max
dt = 1.0 / Fs
t = np.arange(0, 2.0, dt)

s1 = np.sin(Omega * t) + (1.0 / 3.0) * np.cos(3 * Omega * t)

plt.figure(figsize=(8, 3))
plt.plot(t, s1, 'b-', label=r'$s_1(t) = \sin(\Omega t) + \frac{1}{3}\cos(3\Omega t)$')
plt.title('Task i: Signal $s_1(t)$ in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

N1 = len(s1)
freqs1 = np.arange(N1) * (Fs / N1)

s1_dft = compute_dft(s1)
s1_dft_mag = np.sqrt(np.real(s1_dft)**2 + np.imag(s1_dft)**2)
s1_dft_phase = np.arctan2(np.imag(s1_dft), np.real(s1_dft))

s1_dct = compute_dct2(s1)
s1_dst = compute_dst2(s1)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].stem(freqs1[:N1//2], s1_dft_mag[:N1//2], basefmt=" ")
axs[0, 0].set_title('Task ii: Formula DFT Magnitude Spectrum')
axs[0, 0].set_xlabel('Frequency (Hz)')
axs[0, 0].set_ylabel('|DFT|')
axs[0, 0].grid(True)

axs[0, 1].stem(freqs1[:N1//2], s1_dft_phase[:N1//2], basefmt=" ")
axs[0, 1].set_title('Task ii: Formula DFT Phase Spectrum')
axs[0, 1].set_xlabel('Frequency (Hz)')
axs[0, 1].set_ylabel('Phase (rad)')
axs[0, 1].grid(True)

axs[1, 0].stem(np.abs(s1_dct), basefmt=" ")
axs[1, 0].set_title('Task iii: Formula DCT Magnitude Spectrum')
axs[1, 0].set_xlabel('Coefficient Index')
axs[1, 0].set_ylabel('|DCT|')
axs[1, 0].grid(True)

axs[1, 1].stem(np.abs(s1_dst), basefmt=" ")
axs[1, 1].set_title('Task iii: Formula DST Magnitude Spectrum')
axs[1, 1].set_xlabel('Coefficient Index')
axs[1, 1].set_ylabel('|DST|')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

N2 = 32
n = np.arange(0, N2)
s2 = (0.5)**n

s2_dft = compute_dft(s2)
s2_dct = compute_dct2(s2)
s2_dst = compute_dst2(s2)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].stem(n, np.abs(s2_dft), basefmt=" ")
axs[0, 0].set_title('Task v: DFT Magnitude Spectrum of $s_2[n]$')
axs[0, 0].set_xlabel('k')
axs[0, 0].set_ylabel('|DFT|')
axs[0, 0].grid(True)

axs[0, 1].stem(n, np.arctan2(np.imag(s2_dft), np.real(s2_dft)), basefmt=" ")
axs[0, 1].set_title('Task v: DFT Phase Spectrum of $s_2[n]$')
axs[0, 1].set_xlabel('k')
axs[0, 1].set_ylabel('Phase (rad)')
axs[0, 1].grid(True)

axs[1, 0].stem(n, np.abs(s2_dct), basefmt=" ")
axs[1, 0].set_title('Task v: DCT Magnitude Spectrum of $s_2[n]$')
axs[1, 0].set_xlabel('k')
axs[1, 0].set_ylabel('|DCT|')
axs[1, 0].grid(True)

axs[1, 1].stem(n, np.abs(s2_dst), basefmt=" ")
axs[1, 1].set_title('Task v: DST Magnitude Spectrum of $s_2[n]$')
axs[1, 1].set_xlabel('k')
axs[1, 1].set_ylabel('|DST|')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

def retain_top_k(coeff, percentage):
    k = max(1, int(np.round(len(coeff) * (percentage / 100.0))))
    idx = np.argsort(np.abs(coeff))[::-1][:k]
    mask = np.zeros_like(coeff, dtype=bool)
    mask[idx] = True
    truncated = np.zeros_like(coeff)
    truncated[mask] = coeff[mask]
    return truncated

percentages = [90, 75, 45]
mse_results = {'DFT': [], 'DCT': [], 'DST': []}

for p in percentages:

    dft_trunc = retain_top_k(s2_dft, p)
    s2_rec_dft = compute_idft(dft_trunc)
    mse_dft = np.mean((s2 - s2_rec_dft)**2)
    mse_results['DFT'].append(mse_dft)

    dct_trunc = retain_top_k(s2_dct, p)
    s2_rec_dct = compute_idct2(dct_trunc)
    mse_dct = np.mean((s2 - s2_rec_dct)**2)
    mse_results['DCT'].append(mse_dct)

    dst_trunc = retain_top_k(s2_dst, p)
    s2_rec_dst = compute_idst2(dst_trunc)
    mse_dst = np.mean((s2 - s2_rec_dst)**2)
    mse_results['DST'].append(mse_dst)

print("--- Mean Squared Error (MSE) Results (Formula-Based) ---")
for i, p in enumerate(percentages):
    print(f"Retained {p}% Coefficients:")
    print(f"  IDFT MSE: {mse_results['DFT'][i]:.6e}")
    print(f"  IDCT MSE: {mse_results['DCT'][i]:.6e}")
    print(f"  IDST MSE: {mse_results['DST'][i]:.6e}")