# train.py
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import config
import data_loader
import model_builder


def main():
    print("--- STARTING RF FINGERPRINTING PROJECT ---")
    
    # 1. Load Data
    X_train, X_val, Y_train, Y_val = data_loader.load_and_preprocess_data()
    
    # 2. Build Model
    model = model_builder.create_cnn_model()
    # try:
    #     print("[Transfer Learning] Loading weights from models/efficient_rf_model.keras...")
    #     model.load_weights('models/efficient_rf_model.keras')
    #     print("Weights loaded successfully!")
    # except Exception as e:
        # print(f"Could not load weights: {e}. Starting from scratch.")
    try:
        model.summary(line_length=100)
    except ValueError:
        print("[Warning] Could not print model summary due to console width.")
    
    # --- NEW: DEFINE CALLBACKS ---
    # A. Reduce LR: If val_loss doesn't improve for 10 epochs, cut LR in half.
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10, 
        min_lr=0.000001, 
        verbose=1
    )

    # B. Checkpoint: Always save the best model found so far (prevents losing progress).
    checkpoint = ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # 3. Train (WITH CALLBACKS)
    print(f"\n[Train] Starting training for {config.EPOCHS} epochs...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[lr_scheduler, checkpoint],  # <--- CRITICAL ADDITION
        verbose=1
    )
    
    # 4. Save Final Model
    model.save(config.MODEL_SAVE_PATH)
    print(f"\n[Output] Final model saved to {config.MODEL_SAVE_PATH}")
    
    # 5. Plot Results
    plot_history(history)

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.savefig(config.PLOT_SAVE_PATH)
    print(f"[Output] Training plot saved to {config.PLOT_SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    main()
    
