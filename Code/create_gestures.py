import cv2
import numpy as np
import pickle
import os
import random

# Global setting for the final image size. All images will be 50x50.
image_x, image_y = 50, 50

def init_folders():
    """Creates the main 'gestures' directory if it doesn't exist."""
    if not os.path.exists("gestures"):
        print("Creating 'gestures' directory...")
        os.mkdir("gestures")

def store_images(gesture_name):
    """
    Captures and saves 2000 images for a specific gesture.
    """
    total_pics = 2000
    
    # Try to open the main webcam (0) or a secondary one (1)
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    # This is the (x, y) coordinate and (width, height) of the green "Region of Interest" (ROI) box
    x, y, w, h = 300, 100, 300, 300 

    # Create the specific folder for this gesture (e.g., "gestures/scroll_up")
    folder_path = os.path.join("gestures", gesture_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # This lets you *add* images to a folder later without starting over.
    # It finds the highest numbered image (e.g., 1200.jpg) and starts from 1201.
    try:
        existing_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        if existing_images:
            pic_no = max([int(f.split('.')[0]) for f in existing_images])
            print(f"Resuming capture. Starting from image {pic_no + 1}.")
        else:
            pic_no = 0
    except Exception as e:
        print(f"Error checking existing files: {e}")
        pic_no = 0
    
    # Flags to control the capture loop
    flag_start_capturing = False
    frames = 0
    
    while True:
        # Read one frame from the webcam
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        # Flip the image horizontally (so it's like a mirror)
        img = cv2.flip(img, 1)
        
        # --- This is our 'Skin Detector' ---
        # It's better than HSV for bright rooms.
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        skin_cr_min = 133
        skin_cr_max = 173
        skin_cb_min = 77
        skin_cb_max = 127
        
        # This is the "magic" line. It creates a black-and-white mask
        # where only pixels in our skin-color range are white.
        skin_mask = cv2.inRange(
            img_ycrcb, 
            np.array([0, skin_cr_min, skin_cb_min]), 
            np.array([255, skin_cr_max, skin_cb_max])
        )
        
        # Clean up the mask to remove small white dots (noise)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        skin_mask = cv2.filter2D(skin_mask, -1, disc)
        blur = cv2.GaussianBlur(skin_mask, (11, 11), 0)
        thresh = cv2.medianBlur(blur, 15)
        # --- End of Skin Detection ---
        
        # Crop the black-and-white mask to *only* the green box
        thresh_roi = thresh[y:y+h, x:x+w]
        
        # Find the outlines of all white shapes in the box
        contours_data = cv2.findContours(thresh_roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # This line makes it compatible with all OpenCV versions
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

        if len(contours) > 0:
            # Find the biggest white shape (which should be your hand)
            contour = max(contours, key=cv2.contourArea)
            
            # Start saving images if 'c' was pressed and the hand is big enough
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                
                # Crop the *exact* hand shape from the mask
                save_img = thresh_roi[y1:y1+h1, x1:x1+w1]
                
                # This makes the image square, so we don't 'squish' the gesture
                # It adds black bars to the top/bottom or left/right.
                if w1 > h1:
                    padding = int((w1 - h1) / 2)
                    save_img = cv2.copyMakeBorder(save_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                else:
                    padding = int((h1 - w1) / 2)
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, (0, 0, 0))
                
                # Resize to our final 50x50 size
                save_img = cv2.resize(save_img, (image_x, image_y))
                
                # Randomly flip some images (simple data augmentation)
                if random.randint(0, 10) % 2 == 0:
                    save_img = cv2.flip(save_img, 1)
                
                # Save the final 50x50 image
                save_path = os.path.join(folder_path, f"{pic_no}.jpg")
                cv2.imwrite(save_path, save_img)
                
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))

        # Draw the green ROI box on the main camera feed
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Show the number of pictures taken
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        # Show the main camera
        cv2.imshow("Capturing gesture", img)
        # Show the black-and-white "what the computer sees" window
        cv2.imshow("Thresh", thresh_roi)
        
        keypress = cv2.waitKey(1) & 0xFF
        
        if keypress == ord('c'):
            # Toggle capturing on/off
            flag_start_capturing = not flag_start_capturing
            frames = 0
            
        if flag_start_capturing:
            frames += 1 # This adds a 50-frame delay so you can get your hand ready
            
        if pic_no == total_pics:
            print(f"Captured {total_pics} images. Finished.")
            break
            
        if keypress == ord('q'):
            print("Quitting...")
            break

    cam.release()
    cv2.destroyAllWindows()

# --- This is the code that runs when you execute the script ---
if __name__ == "__main__":
    init_folders()
    # Get the name for the folder (e.g., "scroll_up")
    g_name = input("Enter gesture name (e.g., 'app_switch', 'scroll_up'): ").strip().lower()
    # Make the name safe for a folder
    g_name = g_name.replace(' ', '_').replace('.', '').replace('/', '')
    
    if not g_name:
        print("Gesture name cannot be empty. Exiting.")
    else:
        print(f"Preparing to capture images for gesture: '{g_name}'")
        print("Press 'c' to start/stop capturing. Press 'q' to quit.")
        store_images(g_name)