import cv2
from glob import glob
import numpy as np
from sklearn.utils import shuffle
import pickle
import os

def get_label_map():
    """
    Finds all unique gesture folders and creates a mapping
    from the folder name (e.g., 'fist') to an integer (e.g., 0).
    It also saves this map to a file for later use.
    """
    # Find all subdirectories in the 'gestures' folder
    gesture_folders = glob("gestures/*")
    
    # Get the base name of each folder (e.g., 'fist')
    labels = [os.path.basename(folder) for folder in gesture_folders if os.path.isdir(folder)]
    labels.sort() # Sort for consistent mapping
    
    # Create the dictionary mapping
    # e.g., {'fist': 0, 'peace_sign': 1, 'thumbs_up': 2}
    label_map = {label: i for i, label in enumerate(labels)}
    
    # Save this map. The prediction script will need it!
    with open("label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)
        
    print(f"Found {len(labels)} gestures. Mapping created:")
    print(label_map)
    
    return label_map

def pickle_images_labels(label_map):
    """
    Loads all images, converts them to grayscale numpy arrays,
    and pairs them with their correct integer label.
    """
    images_labels = []
    
    # Find all .jpg files in all subfolders of 'gestures'
    images_paths = glob("gestures/*/*.jpg")
    images_paths.sort()
    
    print(f"Loading {len(images_paths)} images...")
    
    for image_path in images_paths:
        # Extract the string label (the folder name)
        # e.g., 'gestures/fist/1.jpg' -> 'fist'
        try:
            label_str = image_path[image_path.find(os.sep)+1 : image_path.rfind(os.sep)]
        except Exception as e:
            print(f"Error parsing path {image_path}: {e}")
            continue
            
        # Use the map to get the integer label
        label_int = label_map[label_str]
        
        # Read the image in grayscale (0 flag)
        img = cv2.imread(image_path, 0)
        
        # Append the tuple (image_array, integer_label)
        images_labels.append((np.array(img, dtype=np.uint8), label_int))
        
    return images_labels

# --- Main Execution ---

# 1. Create and save the label-to-integer mapping
label_map = get_label_map()

# 2. Load all images and apply integer labels
images_labels = pickle_images_labels(label_map)

# 3. Shuffle the dataset
# (One shuffle is sufficient, no need for four)
images_labels = shuffle(images_labels)

# 4. Unzip the list of tuples into two separate lists
images, labels = zip(*images_labels)
print(f"Total images & labels loaded: {len(images_labels)}")

# 5. Define the split points
total_count = len(images)
train_split = int(5/6 * total_count)
test_split = int(11/12 * total_count)

# 6. Split the data
print("Splitting dataset...")
# 
# Train Set: First 5/6 (83.3%)
train_images = images[:train_split]
train_labels = labels[:train_split]

# Test Set: Next 1/12 (8.3%)
test_images = images[train_split:test_split]
test_labels = labels[train_split:test_split]

# Validation Set: Last 1/12 (8.3%)
val_images = images[test_split:]
val_labels = labels[test_split:]

print(f"Training set:   {len(train_images)} images")
print(f"Testing set:    {len(test_images)} images")
print(f"Validation set: {len(val_images)} images")

# 7. Save all the processed data to files
print("Saving processed data to files...")

with open("train_images", "wb") as f:
    pickle.dump(train_images, f)
with open("train_labels", "wb") as f:
    pickle.dump(train_labels, f)
del train_images, train_labels # Clear memory

with open("test_images", "wb") as f:
    pickle.dump(test_images, f)
with open("test_labels", "wb") as f:
    pickle.dump(test_labels, f)
del test_images, test_labels # Clear memory

with open("val_images", "wb") as f:
    pickle.dump(val_images, f)
with open("val_labels", "wb") as f:
    pickle.dump(val_labels, f)
del val_images, val_labels # Clear memory

print("Data processing complete. All files saved.")