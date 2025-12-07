# model_builder.py (The "Super-Model" ResNet Version)
from tensorflow.keras import layers, models, optimizers, Input
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, MaxPooling1D
import config

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # First Conv
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second Conv
    x = Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Adjust shortcut if dimensions change (stride > 1 or filters change)
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    # Add (Skip Connection)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def create_cnn_model():
    print("[Model] Building Efficient-ResNet (Lightweight)...")
    
    inputs = Input(shape=(config.IQ_LEN, 2))
    
    # 1. Initial Feature Extractor (Stride 2 to reduce length immediately)
    x = Conv1D(32, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Shape: (4096, 32)
    
    # 2. ResNet Stack
    # Stage 1: 32 filters (Maintain size)
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    
    # Stage 2: 64 filters (Downsample)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)
    # Shape: (2048, 64)
    
    # Stage 3: 128 filters (Downsample)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    # Shape: (1024, 128)
    
    # Stage 4: 256 filters (Downsample) - Optional, helps deeper features
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    # Shape: (512, 256)

    # 3. Global Features
    x = GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    
    # 4. Classifier
    # Add a small dense layer to mix features before classification
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    
    outputs = Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
