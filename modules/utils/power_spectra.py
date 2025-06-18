import numpy as np
import os
import matplotlib.pyplot as plt

## 1D - Signal
def compute_1d_power_spectrum(signal):
    
    f = np.fft.rfft(signal)
    power = np.abs(f) ** 2
    freqs = np.fft.rfftfreq(len(signal))
    return freqs, power

def relative_error(sig1, sig2):
    
    norm_diff = np.linalg.norm(sig1 - sig2)
    norm_ref = np.linalg.norm(sig2)
    return norm_diff / (norm_ref + 1e-8)

def plot_1d_power_spectra_comparison(noisy, true_signal, reconstructed):
    
    signals = {
        'Noisy Signal': noisy,
        'True Signal': true_signal,
        'Reconstructed Signal': reconstructed
    }

    fig, axes = plt.subplots(1, 1, figsize=(8, 4), sharex=False)

    # --- 2. Plot power spectra ---
    for label, sig in signals.items():
        freqs, power = compute_1d_power_spectrum(sig)
        axes.plot(freqs[1:], power[1:], label=label, alpha=0.8)  # skip DC
    axes.set_title('Power Spectrum (Log)')
    axes.set_xlabel('Frequency')
    axes.set_ylabel('Power')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.legend()
    axes.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 3. Print error ---
    err_recon = relative_error(reconstructed, true_signal)
    print(f"Relative Error (Reconstructed vs True Signal): {err_recon:.4f}")

    plt.tight_layout()
    plt.show()



## 2D ->  1D Radial Power Spectrum
def radial_average(power_spectrum):
    """Compute radially averaged 1D power spectrum."""

    y, x = np.indices(power_spectrum.shape)
    center = np.array(power_spectrum.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)

    # bin by radius
    tbin = np.bincount(r.ravel(), weights=power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_prof = tbin / (nr + 1e-8)
    return radial_prof

def compute_power_spectrum(image):
    """Compute 2D power spectrum of an image using np.fft."""
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2
    return power

def plot_power_spectra_comparison(noisy_image, true_signal, reconstructed_signal):
    """Plots 1D radial power spectra and log-scale relative errors."""
    
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
    true_signal = cv2.cvtColor(true_signal, cv2.COLOR_BGR2GRAY)
    reconstructed_signal = cv2.cvtColor(reconstructed_signal, cv2.COLOR_BGR2GRAY)
    
    spectra = {
        "Noisy Image": noisy_image,
        "True Signal": true_signal,
        "Reconstructed Signal": reconstructed_signal,
    }

    # Compute 1D radial power spectra
    radial_spectra = {
        name: radial_average(compute_power_spectrum(img)) for name, img in spectra.items()
    }

    # Compute relative error (in radial spectrum domain)
    rel_error_signal = np.abs(radial_spectra["Reconstructed Signal"] - radial_spectra["True Signal"]) / \
                       (np.abs(radial_spectra["True Signal"]) + 1e-8)
                       
    # Plot 1D power spectra
    fig, axs = plt.subplots(2, 1, figsize=(7, 8))
    
    for name, spectrum in radial_spectra.items():
        axs[0].plot(np.log1p(spectrum), label=name)
    
    axs[0].set_title("Radially Averaged Power Spectra (log-scale)")
    axs[0].set_xlabel("Radial Frequency")
    axs[0].set_ylabel("log(1 + Power)")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot relative error (log scale)
    axs[1].plot(np.log1p(rel_error_signal), label="Rel Error: Reconstructed vs True Signal", color='blue')
    axs[1].set_title("Relative Error in Power Spectrum (log-scale)")
    axs[1].set_xlabel("Radial Frequency")
    axs[1].set_ylabel("log(1 + Relative Error)")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    print('running __power_spectra.py__')