import cv2
import numpy as np
import pickle
import os
import random

# Global setting for the final image size
image_x, image_y = 50, 50

def get_hand_hist():
    """Loads the pre-calculated hand histogram file."""
    # This file MUST exist. Run the set_hand_histogram.py script first.
    if not os.path.exists("hist"):
        print("ERROR: 'hist' file not found.")
        print("Please run the 'set_hand_histogram.py' script first to calibrate.")
        exit()
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def init_folders():
    """Creates the main 'gestures' directory if it doesn't exist."""
    if not os.path.exists("gestures"):
        print("Creating 'gestures' directory...")
        os.mkdir("gestures")

def store_images(gesture_name):
    """
    Captures and saves images for a specific gesture.
    Images will be saved in a subfolder named after the gesture.
    """
    total_pics = 1200
    hist = get_hand_hist()
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    x, y, w, h = 300, 100, 300, 300 # The green ROI box

    # Create the specific folder for this gesture
    folder_path = os.path.join("gestures", gesture_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Check for existing images to avoid overwriting
    # This lets you add more images to a gesture later.
    try:
        existing_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        if existing_images:
            # Find the highest numbered image (e.g., '1200.jpg')
            pic_no = max([int(f.split('.')[0]) for f in existing_images])
            print(f"Resuming capture. Starting from image {pic_no + 1}.")
        else:
            pic_no = 0
    except Exception as e:
        print(f"Error checking existing files: {e}")
        pic_no = 0
    
    flag_start_capturing = False
    frames = 0
    
    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Perform Back-Projection (find hand color)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        
        # 2. Clean up the mask
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        
        # 3. Create a binary thresholded image
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 4. Crop the thresholded image to the ROI
        thresh_roi = thresh[y:y+h, x:x+w]
        
        # 5. Find contours
        # --- FIX: Changed [1] to [0] for modern OpenCV (4.x) ---
        # OpenCV 3.x returns (img, contours, hierarchy)
        # OpenCV 4.x returns (contours, hierarchy)
        contours_data = cv2.findContours(thresh_roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # This line makes it compatible with both OpenCV 3 and 4
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            
            # Start capturing if contour is large enough and we've waited
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                
                # Crop the hand from the thresholded ROI
                save_img = thresh_roi[y1:y1+h1, x1:x1+w1]
                
                # 6. Make the image square by padding
                if w1 > h1 :
                     # Pad top/bottom
                    padding = int((w1 - h1) / 2)
                    save_img = cv2.copyMakeBorder(save_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                else:
                    # Pad left/right
                    padding = int((h1 - w1) / 2)
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, (0, 0, 0))
                
                # 7. Resize to final size
                save_img = cv2.resize(save_img, (image_x, image_y))
                
                # 8. Data Augmentation (simple flip)
                if random.randint(0, 10) % 2 == 0:
                    save_img = cv2.flip(save_img, 1)
                
                # 9. Save the final image
                save_path = os.path.join(folder_path, f"{pic_no}.jpg")
                cv2.imwrite(save_path, save_img)
                
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))

        # --- Display Windows ---
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", img)
        cv2.imshow("Thresh", thresh_roi) # Show the cropped threshold
        
        # --- Keypress Logic ---
        keypress = cv2.waitKey(1) & 0xFF
        
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing # Toggle capturing
            frames = 0 # Reset frame count on toggle
            
        if flag_start_capturing:
            frames += 1
            
        if pic_no == total_pics:
            print(f"Captured {total_pics} images. Finished.")
            break
            
        if keypress == ord('q'):
            print("Quitting...")
            break

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()

# --- Main execution ---
if __name__ == "__main__":
    init_folders()
    
    # Get gesture name from user
    g_name = input("Enter gesture name (e.g., 'fist', 'peace_sign'): ").strip().lower()
    
    # Sanitize the name to be a valid folder name
    g_name = g_name.replace(' ', '_').replace('.', '').replace('/', '')
    
    if not g_name:
        print("Gesture name cannot be empty. Exiting.")
    else:
        print(f"Preparing to capture images for gesture: '{g_name}'")
        print("Press 'c' to start/stop capturing. Press 'q' to quit.")
        store_images(g_name)