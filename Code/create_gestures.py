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
    # --- THIS IS THE CHANGE YOU REQUESTED ---
    total_pics = 2000
    # --- END OF CHANGE ---
    
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
    
    flag_start_capturing = False
    frames = 0
    
    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        thresh_roi = thresh[y:y+h, x:x+w]
        
        contours_data = cv2.findContours(thresh_roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh_roi[y1:y1+h1, x1:x1+w1]
                
                if w1 > h1:
                    padding = int((w1 - h1) / 2)
                    save_img = cv2.copyMakeBorder(save_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                else:
                    padding = int((h1 - w1) / 2)
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, (0, 0, 0))
                
                save_img = cv2.resize(save_img, (image_x, image_y))
                
                if random.randint(0, 10) % 2 == 0:
                    save_img = cv2.flip(save_img, 1)
                
                save_path = os.path.join(folder_path, f"{pic_no}.jpg")
                cv2.imwrite(save_path, save_img)
                
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", img)
        cv2.imshow("Thresh", thresh_roi)
        
        keypress = cv2.waitKey(1) & 0xFF
        
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            frames = 0
            
        if flag_start_capturing:
            frames += 1
            
        if pic_no == total_pics:
            print(f"Captured {total_pics} images. Finished.")
            break
            
        if keypress == ord('q'):
            print("Quitting...")
            break

    cam.release()
    cv2.destroyAllWindows()

# --- Main execution ---
if __name__ == "__main__":
    init_folders()
    g_name = input("Enter gesture name (e.g., 'fist', 'peace_sign'): ").strip().lower()
    g_name = g_name.replace(' ', '_').replace('.', '').replace('/', '')
    
    if not g_name:
        print("Gesture name cannot be empty. Exiting.")
    else:
        print(f"Preparing to capture images for gesture: '{g_name}'")
        print("Press 'c' to start/stop capturing. Press 'q' to quit.")
        store_images(g_name)