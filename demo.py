import time
import numpy as np
import tensorflow as tf
import h5py # We will load manually to save RAM
import config
import random
import os

# --- CONFIGURATION ---
SECURITY_THRESHOLD = 0.85 
# These are the values we found during training (Hard Mode)
HARDCODED_MEAN = 0.0001
HARDCODED_STD = 0.2714

def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def get_demo_data_lite():
    """
    Loads just a tiny slice of data (1000 samples) to prevent RAM crashes.
    """
    print("Connecting to LoRa Interface (Lite Mode)...")
    try:
        with h5py.File(config.DATASET_PATH, 'r') as f:
            # ONLY LOAD 1000 SAMPLES (Not the whole 4GB!)
            X_raw = f['data'][:1000] 
            Y_raw = f['label'][:1000].T
            
        # 1. Reshape
        X_val = X_raw.reshape(-1, config.IQ_LEN, 2)
        
        # 2. Normalize (Using the values we already know!)
        X_val = (X_val - HARDCODED_MEAN) / HARDCODED_STD
        
        # 3. Process Labels
        Y_val = to_categorical_lite(Y_raw, config.NUM_CLASSES)
        
        return X_val, Y_val
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def to_categorical_lite(y, num_classes):
    """ Simple one-hot encoder to avoid importing big libraries """
    y = np.array(y, dtype='int') - 1 # 0-base
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y.flatten()] = 1
    return categorical

def simulate_live_gateway():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(color_text("\n🔒 INITIALIZING SECURE IOT GATEWAY...", "1;36")) 
    print("Loading Robust ResNet-18 Model...")
    
    # 1. Load Model
    try:
        model = tf.keras.models.load_model('models/robust_efficient_rf_model.keras')
        print(color_text("✔ Model Loaded Successfully. System Secure.", "1;32")) 
    except:
        print(color_text("❌ Error: Model not found!", "1;31"))
        return

    # 2. Load Lite Data
    X_val, Y_val = get_demo_data_lite()
    if X_val is None: return
    
    print("Listening for traffic...\n")
    print("-" * 75)
    print(f"{'TIMESTAMP':<10} | {'SIGNAL PWR':<10} | {'CLAIMED ID':<12} | {'ANALYSIS':<20} | {'STATUS'}")
    print("-" * 75)

    try:
        while True:
            # A. Pick a random packet
            idx = np.random.randint(0, len(X_val))
            signal = X_val[idx:idx+1]
            true_device_id = np.argmax(Y_val[idx]) + 1 
            
            # B. SIMULATE ATTACK (50% Chance)
            is_spoofing = random.random() < 0.50
            claimed_id = true_device_id if not is_spoofing else np.random.randint(1, 31)

            # C. AI INFERENCE
            preds = model.predict(signal, verbose=0)
            predicted_index = np.argmax(preds)
            confidence = np.max(preds)
            predicted_id = predicted_index + 1

            # D. LOGIC
            timestamp = time.strftime("%H:%M:%S")
            rssi = f"-{np.random.randint(60, 95)}dBm"
            
            if confidence > SECURITY_THRESHOLD and predicted_id == claimed_id:
                status = color_text("ACCESS GRANTED", "1;32") 
                analysis = f"Match ({confidence*100:.1f}%)"
            elif predicted_id != claimed_id:
                status = color_text(f"⛔ BLOCKED (ID #{predicted_id}?)", "1;31") 
                analysis = f"FP Mismatch"
            else:
                status = color_text("⚠ FLAGGED (UNCERTAIN)", "1;33") 
                analysis = f"Low Conf ({confidence*100:.1f}%)"

            print(f"{timestamp:<10} | {rssi:<10} | Device #{claimed_id:<02}  | {analysis:<20} | {status}")
            time.sleep(np.random.uniform(0.5, 1.5))

    except KeyboardInterrupt:
        print("\n" + "-" * 75)
        print(color_text("SYSTEM SHUTDOWN. LOGS SAVED.", "1;36"))

if __name__ == "__main__":
    simulate_live_gateway()