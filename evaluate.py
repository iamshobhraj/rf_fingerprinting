import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import data_loader
import config

def evaluate():
    print("--- EVALUATING MODEL PERFORMANCE ---")
    
    # 1. Load Data (Validation Set only)
    print("Loading Validation Data...")
    _, X_val, _, Y_val = data_loader.load_and_preprocess_data()
    
    # 2. Load the Efficient Model
    model_path = 'models/robust_efficient_rf_model.keras'
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return
    
    # 3. Predict
    print("Running predictions...")
    Y_pred_probs = model.predict(X_val, verbose=1)
    Y_pred = np.argmax(Y_pred_probs, axis=1) # Convert probabilities to Class IDs
    
    # Convert True Labels from One-Hot back to Integers
    Y_true = np.argmax(Y_val, axis=1)
    
    # 4. Generate Report
    print("\n--- CLASSIFICATION REPORT ---")
    # This shows Precision/Recall for EACH device
    print(classification_report(Y_true, Y_pred))
    
    # 5. Plot Confusion Matrix
    print("Generating Confusion Matrix Plot...")
    cm = confusion_matrix(Y_true, Y_pred)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='viridis', fmt='d')
    plt.title('RF Fingerprinting Confusion Matrix (30 Devices)')
    plt.xlabel('Predicted Device ID')
    plt.ylabel('Actual Device ID')
    
    # Save it for your report
    plt.savefig('plots/confusion_matrix.png')
    print("Saved matrix to plots/confusion_matrix.png")
    
    # Check if we are in a headless environment
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    evaluate()
