import numpy as np
import pickle
import cv2
import os
from glob import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam  # Import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
# --- MODIFICATION: Import the new callbacks ---
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# New Imports for Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set TensorFlow logging to be less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Set data format to 'channels_last' (e.g., 50, 50, 1)
K.set_image_data_format('channels_last')

def get_num_of_classes():
    """Counts the number of gesture subfolders."""
    return len(glob('gestures/*'))

def load_label_map():
    """Loads the label map and creates a class names list."""
    if not os.path.exists("label_map.pkl"):
        print("ERROR: 'label_map.pkl' file not found.")
        print("Please run the 'load_images.py' script first.")
        exit()
        
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    
    class_names = [name for name, index in sorted(label_map.items(), key=lambda item: item[1])]
    return class_names

def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    """Generates and saves a confusion matrix plot."""
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred_classes)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt=".2f",
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")

def cnn_model(image_x, image_y, num_of_classes):
    """
    Defines the new, deeper CNN architecture with Batch Normalization.
    """
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(32, (3,3), input_shape=(image_y, image_x, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Block 2
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Block 3
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Classifier Head
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_of_classes, activation='softmax'))

    # Compile the model
    # --- MODIFICATION: Use Adam object with a starting learning rate ---
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Checkpoint: This saves the *best* version of the model
    filepath = "cnn_model_keras.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    # --- MODIFICATION: Add the Learning Rate Scheduler ---
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', # Watch the validation loss
        factor=0.2,         # Reduce LR by 80% (0.001 -> 0.0002)
        patience=3,         # Wait 3 epochs with no improvement before dropping
        verbose=1,
        min_lr=1e-6         # Don't drop below this
    )
    
    # Add both callbacks to the list
    callbacks_list = [checkpoint1, lr_scheduler]

    return model, callbacks_list

def train():
    print("Loading pre-processed data...")
    
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f)) / 255.0
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f)) / 255.0
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)
        
    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f)) / 255.0
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    image_y, image_x = train_images.shape[1], train_images.shape[2]
    num_of_classes = get_num_of_classes()
    class_names = load_label_map()

    print(f"Image dimensions: {image_x}x{image_y}")
    print(f"Found {num_of_classes} classes: {class_names}")

    train_images = np.reshape(train_images, (train_images.shape[0], image_y, image_x, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_y, image_x, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], image_y, image_x, 1))

    train_labels_onehot = to_categorical(train_labels, num_of_classes)
    val_labels_onehot = to_categorical(val_labels, num_of_classes)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    print("Building model...")
    model, callbacks_list = cnn_model(image_x, image_y, num_of_classes)
    model.summary()

    print("Starting training...")
    # Train the model
    model.fit(
        datagen.flow(train_images, train_labels_onehot, batch_size=500),
        validation_data=(val_images, val_labels_onehot),
        # --- MODIFICATION: Increased epochs for scheduler ---
        epochs=40,
        steps_per_epoch=len(train_images) / 500,
        callbacks=callbacks_list # This list now includes the scheduler
    )

    print("Training complete. Loading best saved model for evaluation...")
    
    model = load_model('cnn_model_keras.h5')

    scores = model.evaluate(test_images, to_categorical(test_labels, num_of_classes), verbose=0)
    print(f"\nTest Set Evaluation:")
    print(f"  Test Loss: {scores[0]:.4f}")
    print(f"  Test Accuracy: {scores[1]*100:.2f}%")
    print(f"  Test Error: {100 - scores[1] * 100:.2f}%")
    
    y_pred_probs = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    plot_confusion_matrix(test_labels, y_pred_classes, class_names)

# --- Run the training ---
if __name__ == "__main__":
    train()
    K.clear_session()