import cv2
import os
from glob import glob
import time

def flip_images():
    """
    Performs data augmentation by creating a horizontally flipped
    copy of every image in the 'gestures' subfolders.
    """
    gest_folder = "gestures"
    
    # Get all gesture subfolders (e.g., 'gestures/fist', 'gestures/peace_sign')
    gesture_folders = glob(os.path.join(gest_folder, "*"))
    
    print(f"Found {len(gesture_folders)} gesture folders. Starting augmentation...")
    
    for folder_path in gesture_folders:
        # Check if it's actually a directory
        if not os.path.isdir(folder_path):
            continue
            
        gesture_name = os.path.basename(folder_path)
        
        # Find all .jpg images in this folder
        image_paths = glob(os.path.join(folder_path, "*.jpg"))
        
        if not image_paths:
            print(f"No images found in '{gesture_name}'. Skipping.")
            continue
            
        print(f"Augmenting '{gesture_name}': {len(image_paths)} images found.")
        
        # We need to find the highest existing image number to avoid overwriting
        try:
            # Get all image numbers (e.g., '1.jpg' -> 1)
            numbers = [int(os.path.basename(p).split('.')[0]) for p in image_paths]
            start_index = max(numbers) + 1
        except ValueError:
            print(f"Warning: Could not parse image numbers in '{gesture_name}'.")
            # Fallback: just use the total count as the starting index
            start_index = len(image_paths) + 1
        
        print(f"New images will start from index: {start_index}")
        
        new_image_count = 0
        
        for img_path in image_paths:
            try:
                # Read the original image (in grayscale)
                img = cv2.imread(img_path, 0)
                if img is None:
                    print(f"Warning: Could not read {img_path}. Skipping.")
                    continue
                
                # Flip the image horizontally
                flipped_img = cv2.flip(img, 1)
                
                # Create the new path
                new_img_name = f"{start_index + new_image_count}.jpg"
                new_path = os.path.join(folder_path, new_img_name)
                
                # Save the new flipped image
                cv2.imwrite(new_path, flipped_img)
                new_image_count += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        print(f"Finished '{gesture_name}'. Added {new_image_count} new flipped images.")
        time.sleep(1) # Pause just to make output readable

    print("Data augmentation complete.")

# --- Run the function ---
if __name__ == "__main__":
    flip_images()