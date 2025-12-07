import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the Hard Dataset
FILENAME = 'Train/dataset_training_aug.h5' 

print(f"Inspecting {FILENAME}...")

with h5py.File(FILENAME, 'r') as f:
    # Load 1st signal
    data = f['data'][0] 
    label = f['label'][0]
    
    # Reshape (same logic as training)
    # The file is (16384,), we reshape to (8192, 2)
    signal = data.reshape(-1, 2)
    
    # Extract I and Q
    i_data = signal[:, 0]
    q_data = signal[:, 1]
    
    # Combine for plotting
    complex_signal = i_data + 1j * q_data

    print(f"Label: {label}")
    print(f"Min Value: {np.min(data)}")
    print(f"Max Value: {np.max(data)}")
    print(f"Mean Value: {np.mean(data)}")

    # PLOT 1: Time Domain (Is it too quiet or too loud?)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(i_data[:500], label='I (Real)')
    plt.plot(q_data[:500], label='Q (Imag)', alpha=0.7)
    plt.title(f"Hard Data - Time Domain (First 500 samples)")
    plt.legend()
    plt.grid(True)

    # PLOT 2: Spectrogram (Is the Chirp still there?)
    plt.subplot(2, 1, 2)
    plt.specgram(complex_signal, NFFT=256, Fs=1.0, noverlap=128, cmap='inferno')
    plt.title("Hard Data - Spectrogram (Look for the Chirp!)")
    
    plt.tight_layout()
    plt.show()