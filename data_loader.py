# data_loader.py
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import config  # Import your settings

# data_loader.py

def load_and_preprocess_data():
    print(f"[Data] Loading from {config.DATASET_PATH}...", flush=True)
    
    try:
        with h5py.File(config.DATASET_PATH, 'r') as f:
            X_raw = f['data'][()]
            Y_raw = f['label'][()].T 
    except FileNotFoundError:
        raise FileNotFoundError(f"Check config.py! Could not find {config.DATASET_PATH}")

    # 1. Reshape X
    X_reshaped = X_raw.reshape(-1, config.IQ_LEN, 2)
    
    # --- MEMORY OPTIMIZED NORMALIZATION ---
    print("[Data] Calculating stats...", flush=True)
    mean = np.mean(X_reshaped)
    std = np.std(X_reshaped)
    print(f"   Original Mean: {mean:.4f}, Std: {std:.4f}", flush=True)
    
    print("[Data] Normalizing in-place (this saves RAM)...", flush=True)
    # Modify the array directly ( -= and /= ) instead of creating a new one
    X_reshaped -= mean
    X_reshaped /= std
    
    print(f"   Normalized Mean: {np.mean(X_reshaped):.4f} (Should be ~0)", flush=True)
    # --------------------------------------
    
    # 2. Process Y
    Y_int = Y_raw.astype(int) - 1
    Y_onehot = to_categorical(Y_int, config.NUM_CLASSES)
    
    # 4. Split
    print("[Data] Splitting into Train/Val...", flush=True)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_reshaped, Y_onehot, 
        test_size=config.TEST_SPLIT, 
        random_state=config.RANDOM_SEED,
        stratify=Y_int
    )
    
    return X_train, X_val, Y_train, Y_val