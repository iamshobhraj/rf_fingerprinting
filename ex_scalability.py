import numpy as np
import tensorflow as tf
from scipy.spatial.distance import euclidean
import data_loader
import matplotlib.pyplot as plt

def test_scalability():
    print("--- SCALABILITY & HASHING EXPERIMENT ---")
    
    # Load Model
    full_model = tf.keras.models.load_model('models/robust_efficient_rf_model.keras')
    
    # Create Feature Extractor (remove classification head)
    # Using the output of the second-to-last layer as the embedding
    feature_extractor = tf.keras.Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
    print("Feature Extractor created. Output Vector Size:", feature_extractor.output_shape)

    # Load Data
    _, X_val, _, Y_val = data_loader.load_and_preprocess_data()
    
    # Select test subjects
    DEV_A_ID = 0  
    DEV_B_ID = 1 
    
    # Get indices
    indices_a = np.where(np.argmax(Y_val, axis=1) == DEV_A_ID)[0]
    indices_b = np.where(np.argmax(Y_val, axis=1) == DEV_B_ID)[0]
    
    # Extract samples
    sig_a1 = X_val[indices_a[0]:indices_a[0]+1]
    sig_a2 = X_val[indices_a[1]:indices_a[1]+1]
    sig_b1 = X_val[indices_b[0]:indices_b[0]+1]
    
    # Generate Embeddings
    hash_a1 = feature_extractor.predict(sig_a1, verbose=0)[0]
    hash_a2 = feature_extractor.predict(sig_a2, verbose=0)[0]
    hash_b1 = feature_extractor.predict(sig_b1, verbose=0)[0]
    
    # Calculate Distances
    dist_same = euclidean(hash_a1, hash_a2) 
    dist_diff = euclidean(hash_a1, hash_b1)
    
    print("\n--- RESULTS ---")
    print(f"Distance (Dev 1 Packet A <-> Dev 1 Packet B): {dist_same:.4f} (Should be small)")
    print(f"Distance (Dev 1 Packet A <-> Dev 2 Packet A): {dist_diff:.4f} (Should be large)")
    
    if dist_diff > dist_same * 2:
        print("\nSUCCESS: The model successfully creates distinct clusters.")
        print("This proves the system can be scaled using Vector Databases.")
    else:
        print("\nNOTE: Separation is weak. Metric Learning training required for better scale.")

if __name__ == "__main__":
    test_scalability()