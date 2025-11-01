import cv2
import os
import random
import numpy as np
from glob import glob

def get_image_size():
    """Robustly finds the first available gesture image to get its size."""
    # Find all .jpg files in all subfolders of 'gestures'
    all_images = glob("gestures/*/*.jpg")
    
    if not all_images:
        print("Error: No images found in 'gestures' folders.")
        print("Please run 'create_gestures.py' first.")
        exit()
        
    # Load the first image it finds (in grayscale)
    img = cv2.imread(all_images[0], 0)
    if img is None:
        print(f"Error: Could not read sample image {all_images[0]}")
        exit()
        
    print(f"Detected image size: {img.shape}")
    # img.shape returns (height, width), so we return (y, x)
    return img.shape

def create_montage():
    """
    Creates a grid (montage) of random sample images from each gesture folder.
    """
    try:
        image_y, image_x = get_image_size()
    except TypeError:
        # get_image_size() failed and exited
        return

    # --- 1. Get and sort gesture names alphabetically ---
    gesture_folders = glob("gestures/*")
    # Get just the folder name (e.g., 'fist') and ensure it's a directory
    gestures = [os.path.basename(g) for g in gesture_folders if os.path.isdir(g)]
    gestures.sort() # Sort alphabetically
    
    if not gestures:
        print("No gesture directories found inside 'gestures/'.")
        return

    # --- 2. Set up grid parameters ---
    GRID_COLS = 5
    # Use np.ceil to calculate rows (e.g., 7 gestures / 5 cols = 1.4 -> 2 rows)
    GRID_ROWS = int(np.ceil(len(gestures) / GRID_COLS))
    
    # Calculate the full final image width
    full_width = GRID_COLS * image_x
    
    full_img = None

    print(f"Creating a {GRID_ROWS}x{GRID_COLS} montage...")

    # --- 3. Loop through rows and columns ---
    for i in range(GRID_ROWS):
        col_img = None
        
        # Get the slice of gestures for this row
        begin_index = i * GRID_COLS
        end_index = begin_index + GRID_COLS
        row_gestures = gestures[begin_index:end_index]
        
        for gesture_name in row_gestures:
            # --- 4. Robustly pick a random image ---
            img_list = glob(os.path.join("gestures", gesture_name, "*.jpg"))
            
            if not img_list:
                # If a folder is empty, use a black square
                print(f"Warning: No images found in '{gesture_name}'. Using black square.")
                img = np.zeros((image_y, image_x), dtype=np.uint8)
            else:
                # Pick a random image from the list
                img_path = random.choice(img_list)
                img = cv2.imread(img_path, 0)
                
                # If image is corrupted, use a black square
                if img is None:
                    print(f"Warning: Could not read {img_path}. Using black square.")
                    img = np.zeros((image_y, image_x), dtype=np.uint8)
            
            # --- 5. Stitch images horizontally to build the row ---
            
            if col_img is None:
                col_img = img
            else:
                col_img = np.hstack((col_img, img))
                
        # --- 6. CRITICAL: Pad the last row if it's incomplete ---
        if col_img is None: continue # Skip if this row had no gestures
            
        current_width = col_img.shape[1]
        if current_width < full_width:
            # This is the last row, and it's not full
            padding_width = full_width - current_width
            # Create a black image for the padding
            padding_img = np.zeros((image_y, padding_width), dtype=np.uint8)
            col_img = np.hstack((col_img, padding_img))
            
        # --- 7. Stitch the completed row vertically ---
        
        if full_img is None:
            full_img = col_img
        else:
            full_img = np.vstack((full_img, col_img))

    if full_img is not None:
        print("Montage created successfully.")
        # Save the final montage
        cv2.imwrite('full_img.jpg', full_img)
        
        # Display the image
        cv2.imshow("Gesture Montage (Press any key to close)", full_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not create montage.")

# --- Run the main function ---
if __name__ == "__main__":
    create_montage()