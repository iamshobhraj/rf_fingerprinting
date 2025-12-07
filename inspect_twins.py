import h5py
import numpy as np
import matplotlib.pyplot as plt
import config

def inspect_twins():
    print(f"Loading {config.DATASET_PATH}...")
    with h5py.File(config.DATASET_PATH, 'r') as f:
        X = f['data'][()]
        Y = f['label'][()].T

    # Devices are 1-indexed in label, so Device 6 is 6, Device 21 is 21.
    # (In training we convert to 0-indexed, so 5 and 20. But raw labels are 1-30)
    
    # Extract indices
    idx_dev6 = np.where(Y == 6)[0]
    idx_dev21 = np.where(Y == 21)[0]
    
    print(f"Found {len(idx_dev6)} samples for Device 6")
    print(f"Found {len(idx_dev21)} samples for Device 21")
    
    # Get one sample from each
    sig6 = X[idx_dev6[0]].reshape(-1, 2)
    sig21 = X[idx_dev21[0]].reshape(-1, 2)
    
    # Compute mean signal to see if there's a systematic offset (fingerprint)
    # Averaging over all samples might reveal static fingerprints vs noise
    mean_sig6 = np.mean(X[idx_dev6].reshape(-1, config.IQ_LEN, 2), axis=0)
    mean_sig21 = np.mean(X[idx_dev21].reshape(-1, config.IQ_LEN, 2), axis=0)

    plt.figure(figsize=(15, 10))
    
    # Plot 1: Single Sample Comparison (Time Domain - I Channel)
    plt.subplot(2, 2, 1)
    plt.plot(sig6[:200, 0], label='Device 6 (I)', alpha=0.7)
    plt.plot(sig21[:200, 0], label='Device 21 (I)', alpha=0.7)
    plt.title("Single Sample: I-Channel (First 200 samples)")
    plt.legend()
    
    # Plot 2: Mean Signal Comparison (Time Domain - I Channel)
    plt.subplot(2, 2, 2)
    plt.plot(mean_sig6[:200, 0], label='Mean Dev 6 (I)', linestyle='--')
    plt.plot(mean_sig21[:200, 0], label='Mean Dev 21 (I)', linestyle=':')
    plt.title("Mean Signal: I-Channel (Static Fingerprint?)")
    plt.legend()
    
    # Plot 3: Constellation Density (Visualizing IQ distribution)
    plt.subplot(2, 2, 3)
    plt.hist2d(sig6[:, 0], sig6[:, 1], bins=50, cmap='Blues', range=[[-0.02, 0.02], [-0.02, 0.02]])
    plt.title("Device 6 Constellation")
    
    plt.subplot(2, 2, 4)
    plt.hist2d(sig21[:, 0], sig21[:, 1], bins=50, cmap='Reds', range=[[-0.02, 0.02], [-0.02, 0.02]])
    plt.title("Device 21 Constellation")
    
    plt.tight_layout()
    plt.savefig('plots/twin_inspection.png')
    print("Saved plots/twin_inspection.png")

if __name__ == "__main__":
    inspect_twins()
