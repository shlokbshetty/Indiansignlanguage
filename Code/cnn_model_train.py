import numpy as np
import pickle
import cv2
import os
from glob import glob
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# Set TensorFlow logging to be less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set data format to 'channels_last' (e.g., 50, 50, 1)
K.set_image_data_format('channels_last')

def get_image_size():
    """Robustly finds the first available gesture image to get its size."""
    all_images = glob("gestures/*/*.jpg")
    if not all_images:
        print("Error: No images found in 'gestures' folders.")
        print("Please run 'create_gestures.py' first.")
        exit()
        
    img = cv2.imread(all_images[0], 0)
    if img is None:
        print(f"Error: Could not read sample image {all_images[0]}")
        exit()
    return img.shape

def get_num_of_classes():
    """Counts the number of gesture subfolders."""
    return len(glob('gestures/*'))

def cnn_model(image_x, image_y, num_of_classes):
    """
    Defines the Convolutional Neural Network (CNN) architecture.
    """
    model = Sequential()
    
    # 1. Convolutional Layer: Finds 16 simple features (edges, corners).
    
    model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    
    # 2. Max-Pooling Layer: Reduces the image size, keeping the most important info.
    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # 3. More Layers: Finds more complex features.
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    
    # 4. Flatten Layer: Converts the 2D feature maps into a 1D vector.
    model.add(Flatten())
    
    # 5. Dense Layer: A standard fully-connected "brain" layer.
    model.add(Dense(128, activation='relu'))
    
    # 6. Dropout Layer: Prevents overfitting by randomly "dropping" neurons.
    
    model.add(Dropout(0.2))
    
    # 7. Output Layer: Softmax ensures all outputs add up to 1 (like probabilities).
    model.add(Dense(num_of_classes, activation='softmax'))

    # 8. Compile the model
    # We use 'adam' as it's a very effective, modern optimizer.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 9. Checkpoint: This saves the *best* version of the model during training.
    # --- FIX: Changed monitor from 'val_acc' to 'val_accuracy' ---
    filepath = "cnn_model_keras.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

def train():
    print("Loading pre-processed data...")
    with open("train_images", "rb") as f:
        # --- NEW: Normalize data by dividing by 255.0 ---
        train_images = np.array(pickle.load(f)) / 255.0
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        # --- NEW: Normalize data by dividing by 255.0 ---
        val_images = np.array(pickle.load(f)) / 255.0
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)
        
    # Get image shape and class count from the loaded data
    image_y, image_x = train_images.shape[1], train_images.shape[2]
    num_of_classes = get_num_of_classes()

    print(f"Image dimensions: {image_x}x{image_y}")
    print(f"Found {num_of_classes} classes.")

    # Reshape images to include the 'channel' dimension (1 for grayscale)
    # (count, 50, 50) -> (count, 50, 50, 1)
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))

    # One-Hot Encode the labels
    # e.g., A label of '2' (for 5 classes) becomes [0, 0, 1, 0, 0]
    
    train_labels = to_categorical(train_labels, num_of_classes)
    val_labels = to_categorical(val_labels, num_of_classes)

    print(f"Training data shape: {train_images.shape}")
    print(f"Validation labels shape: {val_labels.shape}")

    print("Building model...")
    model, callbacks_list = cnn_model(image_x, image_y, num_of_classes)
    model.summary()

    print("Starting training...")
    # Train the model
    model.fit(
        train_images, 
        train_labels, 
        validation_data=(val_images, val_labels), 
        epochs=15, 
        batch_size=500, 
        callbacks=callbacks_list
    )

    # Evaluate the *final* model (the checkpoint might be better)
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print(f"Final Model CNN Error: {100 - scores[1] * 100:.2f}%")
    
    # Note: The best model is saved as "cnn_model_keras.h5" by the checkpoint.
    # The 'model.save(...)' line from your original code is redundant.

# --- Run the training ---
if __name__ == "__main__":
    train()
    K.clear_session()
    print("Training complete. Best model saved as 'cnn_model_keras.h5'.")