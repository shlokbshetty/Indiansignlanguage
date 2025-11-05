# This fixes a bug where matplotlib (for plotting) fails to save images
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import pickle
import cv2
import os
from glob import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# Imports for plotting our "report card" graphs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
K.set_image_data_format('channels_last') # (50, 50, 1)

def get_num_of_classes():
    """Counts the number of gesture subfolders."""
    return len(glob('gestures/*'))

def load_label_map():
    """Loads the label map and creates a sorted list of names for the graphs."""
    if not os.path.exists("label_map.pkl"):
        print("ERROR: 'label_map.pkl' file not found.")
        print("Please run the 'load_images.py' script first.")
        exit()
        
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    
    # e.g., ['app_switch', 'close_window', 'nothing', ...]
    class_names = [name for name, index in sorted(label_map.items(), key=lambda item: item[1])]
    return class_names

def plot_training_history(history):
    """Saves a 'learning report' graph showing how the model learned."""
    print("Generating training history graph...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Training history graph saved as 'training_history.png'")

def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    """Saves a 'report card' graph showing where the model got confused."""
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Normalize the matrix to show percentages (0.0 to 1.0)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, 
        annot=True,     # Show the percentages on the map
        fmt=".2f",      # Format to 2 decimal places
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title("Confusion Matrix (Model Report Card)")
    plt.ylabel('True Label (What it actually was)')
    plt.xlabel('Predicted Label (What the model *thought* it was)')
    plt.tight_layout()
    
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")

def cnn_model(image_x, image_y, num_of_classes):
    """
    This defines the architecture of our "brain" (the CNN).
    """
    model = Sequential()
    
    # Block 1: Find simple edges and corners
    model.add(Conv2D(32, (3,3), input_shape=(image_y, image_x, 1), activation='relu', padding='same'))
    model.add(BatchNormalization()) # Helps model train faster
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Block 2: Find more complex shapes (curves, etc.)
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Block 3: Find even more complex features (like fingertips)
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # The "Decision" part of the brain
    model.add(Flatten()) # Turn our 2D maps into one long list
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3)) # Prevents "cheating" by randomly ignoring 30% of neurons
    model.add(Dense(num_of_classes, activation='softmax')) # The final decision layer

    # --- This is our "Assistant Trainer" ---
    
    # The Optimizer defines *how* to learn
    optimizer = Adam(learning_rate=0.0004) # A good, slow learning rate
    
    # The Callbacks watch the model and help it
    
    # 1. Checkpoint: Saves only the *best* version of the model
    filepath = "cnn_model_keras.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    # 2. Learning Rate Scheduler: Slows down learning if the model gets stuck
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', # Watch the "practice exam" score
        factor=0.2,         # If stuck, reduce LR by 80%
        patience=3,         # Wait 3 epochs before reducing
        verbose=1,
        min_lr=1e-6         # Don't go lower than this
    )
    
    callbacks_list = [checkpoint1, lr_scheduler]

    # "Compile" the model, putting all the pieces together
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model, callbacks_list

def train():
    print("Loading pre-processed data...")
    
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f)) / 255.0 # Normalize (0-1)
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f)) / 255.0 # Normalize
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)
        
    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f)) / 255.0 # Normalize
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32) 

    image_y, image_x = train_images.shape[1], train_images.shape[2]
    num_of_classes = get_num_of_classes()
    class_names = load_label_map() # Get names for the plots

    print(f"Image dimensions: {image_x}x{image_y}")
    print(f"Found {num_of_classes} classes: {class_names}")

    # Reshape images to (count, 50, 50, 1) as the model expects
    train_images = np.reshape(train_images, (train_images.shape[0], image_y, image_x, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_y, image_x, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], image_y, image_x, 1))

    # "One-Hot" encode the labels (e.g., '3' -> [0, 0, 0, 1, 0, 0...])
    train_labels_onehot = to_categorical(train_labels, num_of_classes)
    val_labels_onehot = to_categorical(val_labels, num_of_classes)

    # This is our Data Augmentation "machine"
    # It creates new, slightly different images during training
    datagen = ImageDataGenerator(
        rotation_range=10,       # Randomly rotate
        width_shift_range=0.1,   # Randomly shift
        height_shift_range=0.1,
        zoom_range=0.1           # Randomly zoom
    )

    print("Building model...")
    model, callbacks_list = cnn_model(image_x, image_y, num_of_classes)
    model.summary() # Print a summary of the "brain"

    print("Starting training...")
    # This is where the magic happens!
    # We save the 'history' so we can plot it later.
    history = model.fit(
        datagen.flow(train_images, train_labels_onehot, batch_size=500),
        validation_data=(val_images, val_labels_onehot),
        epochs=50, # Train for 50 full cycles
        callbacks=callbacks_list # Use our "assistant trainers"
    )
    
    # --- After training is done, create the plots ---
    plot_training_history(history)

    print("Training complete. Loading best saved model for evaluation...")
    
    # Load the *best* version of the model that was saved
    model = load_model('cnn_model_keras.h5')

    # Run the "final exam" on the test_images
    scores = model.evaluate(test_images, to_categorical(test_labels, num_of_classes), verbose=0)
    print(f"\nTest Set Evaluation:")
    print(f"  Test Loss: {scores[0]:.4f}")
    print(f"  Test Accuracy: {scores[1]*100:.2f}%")
    
    # Get the model's predictions on the "final exam" data
    y_pred_probs = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred_probs, axis=1) # Find the highest probability
    
    # Create the "report card"
    plot_confusion_matrix(test_labels, y_pred_classes, class_names)

# --- This is the code that runs when you execute the script ---
if __name__ == "__main__":
    train()
    K.clear_session() # Clean up memory