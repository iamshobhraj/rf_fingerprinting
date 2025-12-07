import time
import numpy as np
import tensorflow as tf
import data_loader
import config
import random
import os

# --- CONFIGURATION ---
SECURITY_THRESHOLD = 0.85 

def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def simulate_live_gateway():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(color_text("\n🔒 INITIALIZING SECURE IOT GATEWAY...", "1;36")) # Cyan
    print("Loading Robust ResNet-18 Model...")
    
    # 1. Load your 98% Accurate Model
    try:
        model = tf.keras.models.load_model('models/robust_efficient_rf_model.keras')
        print(color_text("✔ Model Loaded Successfully. System Secure.", "1;32")) # Green
    except:
        print(color_text("❌ Error: Model not found! Train it first.", "1;31"))
        return

    # 2. Grab some real data to "replay"
    print("Connecting to LoRa Interface...")
    # We only need X_val (signals) and Y_val (true IDs)
    _, X_val, _, Y_val = data_loader.load_and_preprocess_data()
    print("Listening for traffic...\n")
    
    print("-" * 75)
    print(f"{'TIMESTAMP':<10} | {'SIGNAL PWR':<10} | {'CLAIMED ID':<12} | {'ANALYSIS':<20} | {'STATUS'}")
    print("-" * 75)

    try:
        # Loop to simulate packets arriving
        while True:
            # Select random sample
            idx = np.random.randint(0, len(X_val))
            signal = X_val[idx:idx+1]
            true_device_id = np.argmax(Y_val[idx]) + 1
            
            # Simulate spoofing attack (randomly change claimed ID)
            is_spoofing = random.random() < 0.50
            claimed_id = true_device_id if not is_spoofing else np.random.randint(1, 31)

            # Inference
            preds = model.predict(signal, verbose=0)
            predicted_index = np.argmax(preds)
            confidence = np.max(preds)
            predicted_id = predicted_index + 1

            # Decision Logic
            timestamp = time.strftime("%H:%M:%S")
            rssi = f"-{np.random.randint(60, 95)}dBm"
            
            # Authentication Check
            if confidence > SECURITY_THRESHOLD and predicted_id == claimed_id:
                status = color_text("ACCESS GRANTED", "1;32") # Green
                analysis = f"Match ({confidence*100:.1f}%)"
            
            elif predicted_id != claimed_id:
                status = color_text(f"⛔ BLOCKED (ID #{predicted_id}?)", "1;31") # Red
                analysis = f"FP Mismatch"
                
            else:
                status = color_text("⚠ FLAGGED (UNCERTAIN)", "1;33") # Yellow
                analysis = f"Low Conf ({confidence*100:.1f}%)"

            # Print the log line
            print(f"{timestamp:<10} | {rssi:<10} | Device #{claimed_id:<02}  | {analysis:<20} | {status}")
            
            # Random delay to look like real traffic
            time.sleep(np.random.uniform(0.5, 1.5))

    except KeyboardInterrupt:
        print("\n" + "-" * 75)
        print(color_text("SYSTEM SHUTDOWN. LOGS SAVED.", "1;36"))

if __name__ == "__main__":
    simulate_live_gateway()