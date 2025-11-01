import cv2
import numpy as np
import pickle

def build_squares(frame):
    """
    Draws a 10x5 grid of sampling squares on the frame.
    It also samples the pixels inside these squares and returns them
    as a single, stitched-together image (crop).
    """
    # Define the top-left corner, size, and spacing of the squares
    x, y, w, h = 420, 140, 10, 10
    spacing = 10
    
    # These will hold the pixel data from the squares
    img_crop = None
    crop_rows = None

    for i in range(10):  # 10 rows
        row_pixels = None
        for j in range(5):  # 5 columns
            # Get the small 10x10 pixel region
            sample = frame[y:y+h, x:x+w]
            
            # Stitch this sample to the right of the other samples in its row
            if row_pixels is None:
                row_pixels = sample
            else:
                row_pixels = np.hstack((row_pixels, sample))
            
            # Draw the green rectangle on the main frame
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)
            x += w + spacing
        
        # Stitch this completed row underneath the other rows
        if crop_rows is None:
            crop_rows = row_pixels
        else:
            crop_rows = np.vstack((crop_rows, row_pixels))
        
        # Reset x for the next row and move y down
        img_crop = None
        x = 420
        y += h + spacing
        
    return crop_rows

def capture_and_save_histogram():
    """
    Main function to run the histogram capture application.
    - 'c' captures the histogram from the squares.
    - 's' saves the captured histogram and exits.
    - 'q' quits without saving.
    """
    # State variables
    hist = None
    img_crop = None
    flag_captured = False

    # Open the webcam, trying camera 1 then camera 0
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        # Flip the frame horizontally (like a mirror)
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        
        # Convert the whole frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -----------------------------------------------------------------
        # LOGIC: Before vs. After capturing the histogram
        # -----------------------------------------------------------------

        if not flag_captured:
            # --- STAGE 1: CALIBRATION ---
            # Draw the squares and get the pixel data from them
            img_crop = build_squares(frame)
            
            # Add instruction text
            cv2.putText(frame, "Place hand in squares. Press 'c' to capture.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        else:
            # --- STAGE 2: PREVIEW ---
            # We have a histogram, so now we show a live preview.
            
            # 1. Perform Back-Projection
            # This finds all pixels in the frame that match the colors in our histogram.
            dst = cv2.calcBackProject([hsv_frame], [0, 1], hist, [0, 180, 0, 256], 1)
            
            # 2. Filtering and Thresholding
            # Clean up the back-projection mask to reduce noise.
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 3. Show the cleaned-up mask
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresh", thresh)
            
            # Add instruction text
            cv2.putText(frame, "Calibration done! Press 's' to save and exit.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # -----------------------------------------------------------------
        # Keypress Handling
        # -----------------------------------------------------------------
        
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord('c') and not flag_captured:
            # 'c' was pressed: Capture the histogram
            if img_crop is None:
                print("Error: No sample image to capture from.")
                continue
                
            # Convert the SAMPLED pixels (from the squares) to HSV
            hsv_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
            
            # Calculate the 2D histogram (Hue and Saturation)
            hist = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            
            # Normalize the histogram values to be between 0 and 255
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            
            flag_captured = True  # Flip the state
            print("Histogram captured!")

        elif keypress == ord('s'):
            # 's' was pressed: Save and exit
            if hist is not None:
                # Save the histogram object to a file using pickle
                with open("hist", "wb") as f:
                    pickle.dump(hist, f)
                print("Histogram saved as 'hist'. Exiting.")
                break
            else:
                print("Error: No histogram captured. Press 'c' first.")

        elif keypress == ord('q'):
            # 'q' was pressed: Quit immediately
            print("Exiting without saving.")
            break
            
        # Always display the main camera feed
        cv2.imshow("Set hand histogram", frame)

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()

# --- Run the main function ---
if __name__ == "__main__":
    capture_and_save_histogram()