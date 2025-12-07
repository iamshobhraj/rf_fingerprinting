# config.py

# --- FILE PATHS ---
# Make sure this points to your actual file location
DATASET_PATH = './Train/dataset_training_aug.h5'
MODEL_SAVE_PATH = './models/robust_efficient_rf_model.keras'
PLOT_SAVE_PATH = './plots/training_results.png'

# --- DATA PARAMETERS ---
NUM_CLASSES = 30           # Total devices
INPUT_RAW_LEN = 16384      # Raw length in H5 file
IQ_LEN = INPUT_RAW_LEN // 2 # 8192 (Reshaped length)

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2           # 20% for validation
RANDOM_SEED = 42
