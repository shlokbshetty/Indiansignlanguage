import cv2
from glob import glob
import numpy as np
from sklearn.utils import shuffle
import pickle
import os

def get_label_map():
    """
    Finds all your gesture folders and creates a "dictionary" (label map)
    that maps the name (string) to a number (integer).
    e.g., {'app_switch': 0, 'close_window': 1, ...}
    """
    gesture_folders = glob("gestures/*")
    # Get just the folder names (e.g., 'app_switch')
    labels = [os.path.basename(folder) for folder in gesture_folders if os.path.isdir(folder)]
    labels.sort() # Sort them alphabetically so the map is consistent
    
    if not labels:
        print("ERROR: No gesture folders found in the 'gestures' directory.")
        print("Please run 'create_gestures.py' first.")
        exit()
        
    # Create the dictionary
    label_map = {label: i for i, label in enumerate(labels)}
    
    # Save this dictionary to a file. Our final.py script will need it!
    with open("label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)
        
    print(f"Found {len(labels)} gestures. Mapping created:")
    print(label_map)
    
    return label_map

def pickle_images_labels(label_map):
    """
    This is the "factory." It loads all 18,000+ images from your folders,
    reads them as grayscale, and pairs them with their correct number label.
    """
    images_labels = []
    # Get a list of *all* .jpg files inside *all* gesture folders
    images_paths = glob("gestures/*/*.jpg")
    images_paths.sort()
    
    if not images_paths:
        print("ERROR: No .jpg images found in 'gestures' subfolders.")
        exit()
        
    print(f"Loading {len(images_paths)} images...")
    
    for image_path in images_paths:
        try:
            # e.g., 'gestures/scroll_up/1.jpg' -> 'scroll_up'
            label_str = image_path[image_path.find(os.sep)+1 : image_path.rfind(os.sep)]
        except Exception as e:
            print(f"Error parsing path {image_path}: {e}")
            continue
            
        # Use the dictionary to turn the name 'scroll_up' into a number
        label_int = label_map[label_str]
        
        # Read the image as grayscale (0 flag)
        img = cv2.imread(image_path, 0) 
        
        if img is None:
            print(f"Warning: Could not read {image_path}. Skipping.")
            continue
            
        # Add the (image_data, label_number) pair to our big list
        images_labels.append((np.array(img, dtype=np.uint8), label_int))
        
    return images_labels

# --- This is the code that runs when you execute the script ---

print("Step 1: Creating label map...")
label_map = get_label_map()

print("\nStep 2: Loading and processing images...")
images_labels = pickle_images_labels(label_map)

print("\nStep 3: Shuffling dataset...")
# This is super important! It shuffles all the images randomly
# so the model doesn't just learn one gesture at a time.
images_labels = shuffle(images_labels)

# Unzip the big list of (image, label) pairs into two separate lists
images, labels = zip(*images_labels)
print(f"Total images & labels loaded: {len(images_labels)}")

# 5. Define our 83% / 8.3% / 8.3% split
total_count = len(images)
train_split = int(5/6 * total_count)
test_split = int(11/12 * total_count)

print("\nStep 4: Splitting dataset...")
# The "study notes" (83.3%)
train_images = images[:train_split]
train_labels = labels[:train_split]

# The "final exam" (8.3%)
test_images = images[train_split:test_split]
test_labels = labels[train_split:test_split]

# The "practice quiz" (8.3%)
val_images = images[test_split:]
val_labels = labels[test_split:]

print(f"Training set:   {len(train_images)} images")
print(f"Testing set:    {len(test_images)} images")
print(f"Validation set: {len(val_images)} images")

print("\nStep 5: Saving processed data to files...")
# We save these as pickle files so our trainer can load them fast
with open("train_images", "wb") as f:
    pickle.dump(train_images, f)
with open("train_labels", "wb") as f:
    pickle.dump(train_labels, f)
del train_images, train_labels # Delete from memory to save RAM

with open("test_images", "wb") as f:
    pickle.dump(test_images, f)
with open("test_labels", "wb") as f:
    pickle.dump(test_labels, f)
del test_images, test_labels

with open("val_images", "wb") as f:
    pickle.dump(val_images, f)
with open("val_labels", "wb") as f:
    pickle.dump(val_labels, f)
del val_images, val_labels

print("\nData processing complete. All files saved.")