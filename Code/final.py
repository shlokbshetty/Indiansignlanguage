import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pyttsx3
from threading import Thread
from glob import glob

# --- 1. Setup & Initialization ---

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- FIX: Load the model name we saved in the training script ---
print("Loading model... This may take a moment.")
model = load_model('cnn_model_keras.h5')

# Initialize Text-to-Speech Engine
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
except Exception as e:
    print(f"Warning: pyttsx3 init failed. Voice output will be disabled. Error: {e}")
    engine = None

# Global state for voice
is_voice_on = True

def load_label_map():
    """
    Loads the label map and creates an inverse map to go
    from the model's output index (int) back to a label (str).
    e.g., {0: 'fist', 1: 'peace_sign'}
    """
    if not os.path.exists("label_map.pkl"):
        print("ERROR: 'label_map.pkl' file not found.")
        print("Please run the 'load_images.py' script first.")
        exit()
        
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    
    # Create the inverse map (int -> str)
    # --- FIX: Replaces the need for sqlite3 ---
    inverse_label_map = {v: k for k, v in label_map.items()}
    return inverse_label_map

def get_hand_hist():
    """Loads the pre-calculated hand histogram file."""
    if not os.path.exists("hist"):
        print("ERROR: 'hist' file not found.")
        print("Please run the 'set_hand_histogram.py' script first.")
        exit()
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def get_image_size():
    """Robustly finds the first available gesture image to get its size."""
    # --- FIX: Replaced hardcoded path ---
    all_images = glob("gestures/*/*.jpg")
    if not all_images:
        print("Error: No images found in 'gestures' folders.")
        print("Please run 'create_gestures.py' first.")
        exit()
        
    img = cv2.imread(all_images[0], 0)
    if img is None:
        print(f"Error: Could not read sample image {all_images[0]}")
        exit()
    # img.shape returns (height, width)
    return img.shape[0], img.shape[1]

# --- 2. Load Global Assets ---

# Load the maps, histogram, and image dimensions
inverse_label_map = load_label_map()
hist = get_hand_hist()
image_y, image_x = get_image_size()
print(f"Model loaded. Image size: {image_x}x{image_y}. Found {len(inverse_label_map)} gestures.")

# --- 3. Prediction & Processing Functions ---

def keras_process_image(img):
    """Prepares the cropped image for the Keras model."""
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    
    # --- CRITICAL FIX: Normalize the image ---
    # The model was trained on data divided by 255.0.
    # We MUST do the same for prediction.
    img = img / 255.0
    
    img = np.reshape(img, (1, image_y, image_x, 1))
    return img

def keras_predict(model, image):
    """Gets a prediction from the Keras model."""
    processed = keras_process_image(image)
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = np.argmax(pred_probab)
    return max(pred_probab), pred_class

def get_pred_from_contour(contour, thresh):
    """
    Extracts the gesture from the contour, processes it, and predicts
    using the Keras model.
    """
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    # Crop the hand from the thresholded ROI
    save_img = thresh[y1:y1 + h1, x1:x1 + w1]
    
    # Make the image square by padding
    if w1 > h1:
        padding = int((w1 - h1) / 2)
        save_img = cv2.copyMakeBorder(save_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        padding = int((h1 - w1) / 2)
        save_img = cv2.copyMakeBorder(save_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, (0, 0, 0))
    
    # Get prediction
    pred_probab, pred_class = keras_predict(model, save_img)
    
    # Only return a prediction if confidence is high
    if pred_probab * 100 > 70:
        # --- FIX: Use the inverse_label_map, not the database ---
        text = inverse_label_map[pred_class]
        return text
    
    return ""

def get_img_contour_thresh(img):
    """Applies skin segmentation to find the hand contour."""
    # Define the Region of Interest (ROI)
    x, y, w, h = 300, 100, 300, 300
    
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Skin segmentation using the histogram
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    
    # Filtering and thresholding
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)
    blur = cv2.GaussianBlur(dst, (11, 11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Crop to the ROI
    thresh = thresh[y:y + h, x:x + w]
    
    # Find contours
    # --- NOTE: [0] is correct for modern OpenCV (4.x) ---
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    
    # Draw the green ROI box on the original image
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img, contours, thresh

def say_text(text):
    """Handles text-to-speech in a separate thread."""
    if not is_voice_on or engine is None:
        return
    try:
        while engine._inLoop:
            pass
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"pyttsx3 error: {e}")
        
# --- 4. Application Mode Functions ---

def get_operator(pred_text):
    """
    --- REWRITE ---
    Maps gesture *names* (strings) to operator *symbols* (strings).
    This assumes you have trained gestures with these folder names.
    """
    op_map = {
        "plus": "+",
        "minus": "-",
        "multiply": "*",
        "divide": "/",
        "mod": "%",
        # Add more mappings as needed
    }
    return op_map.get(pred_text, "")

def calculator_mode(cam):
    """Runs the gesture-controlled calculator."""
    global is_voice_on
    
    first_num, second_num, operator = "", "", ""
    calc_text = ""
    info = "Enter first number"
    Thread(target=say_text, args=(info,)).start()
    
    flag_first_num = False
    flag_operator = False
    flag_clear = False
    
    pred_text = ""
    count_same_frames = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img)
        
        old_pred_text = pred_text
        
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                pred_text = get_pred_from_contour(contour, thresh)
            else:
                pred_text = ""
        else:
            pred_text = ""
            
        # --- Logic for holding a gesture ---
        if old_pred_text == pred_text and pred_text != "":
            count_same_frames += 1
        else:
            count_same_frames = 0

        # --- REWRITTEN LOGIC ---
        if count_same_frames > 15: # Hold gesture for ~15 frames
            count_same_frames = 0
            
            op_symbol = get_operator(pred_text)
            
            # 1. Check for CLEAR gesture
            if pred_text == "c":
                first_num, second_num, operator, calc_text = "", "", "", ""
                flag_first_num, flag_operator, flag_clear = False, False, False
                info = "Enter first number"
                Thread(target=say_text, args=(info,)).start()

            # 2. Check for EQUALS gesture
            # (Assuming "best_of_luck_" or "equals" is your equals gesture)
            elif pred_text in ["equals", "best_of_luck_"]:
                if flag_first_num and flag_operator and second_num:
                    try:
                        # Calculate the result
                        calc_text = f"{first_num}{operator}{second_num}"
                        result = str(eval(calc_text))
                        calc_text += f" = {result}"
                        Thread(target=say_text, args=(calc_text,)).start()
                        first_num, second_num, operator = result, "", ""
                        flag_operator = False
                        flag_clear = True # Ready to clear
                    except Exception as e:
                        calc_text = "Error"
                        Thread(target=say_text, args=(calc_text,)).start()
                        
            # 3. Check for an OPERATOR gesture
            elif op_symbol and flag_first_num:
                if not flag_operator:
                    operator = op_symbol
                    calc_text += operator
                    flag_operator = True
                    info = "Enter second number"
                    Thread(target=say_text, args=(info,)).start()
                else: # Already have an operator, chain the calculation
                    try:
                        calc_text = f"{first_num}{operator}{second_num}"
                        result = str(eval(calc_text))
                        first_num = result
                        second_num = ""
                        operator = op_symbol
                        calc_text = f"{first_num}{operator}"
                        Thread(target=say_text, args=(calc_text,)).start()
                    except:
                        calc_text = "Error"
                        
            # 4. Check for a NUMBER gesture
            elif pred_text.isnumeric():
                Thread(target=say_text, args=(pred_text,)).start()
                if flag_clear: # After equals, start new calculation
                    first_num, second_num, operator, calc_text = "", "", "", ""
                    flag_first_num, flag_operator, flag_clear = False, False, False
                    info = "Enter first number"
                
                if not flag_first_num:
                    first_num += pred_text
                    calc_text += pred_text
                elif flag_operator:
                    second_num += pred_text
                    calc_text += pred_text
                    # Auto-set flag_first_num for the *next* operation
                    flag_first_num = True 
            
            # Set this *after* number check
            if first_num and not flag_first_num:
                flag_first_num = True
                info = "Enter operator"
                Thread(target=say_text, args=(info,)).start()

        # --- Display Blackboard ---
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Calculator Mode", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
        cv2.putText(blackboard, "Predicted: " + pred_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        cv2.putText(blackboard, info, (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
        cv2.putText(blackboard, f"Voice: {'ON' if is_voice_on else 'OFF'} (v)", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 127, 0))

        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("thresh", thresh)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q') or keypress == ord('t'):
            break
        elif keypress == ord('v'):
            is_voice_on = not is_voice_on
    
    return 1 if keypress == ord('t') else 0 # 't' for text mode

def text_mode(cam):
    """Runs the gesture-to-text (sign language) mode."""
    global is_voice_on
    text = ""
    word = ""
    count_same_frame = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img)
        
        old_text = text
        
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                text = get_pred_from_contour(contour, thresh)
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                # --- FIX: Use sanitized gesture names ---
                if count_same_frame > 20: # Hold gesture
                    if text in ["i_me", "i", "me"]:
                        if not word: # If word is empty, it's 'I'
                            word += "I "
                            Thread(target=say_text, args=("I",)).start()
                        else: # Otherwise it's 'me'
                            word += "me "
                            Thread(target=say_text, args=("me",)).start()
                    elif text:
                        word = word + text + " "
                        Thread(target=say_text, args=(text, )).start()
                    
                    count_same_frame = 0
            
        elif word != '': # Hand removed, "speak" the word/sentence
            Thread(target=say_text, args=(word, )).start()
            text = ""
            word = ""
            
        # Display Blackboard
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
        cv2.putText(blackboard, "Predicted: " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        cv2.putText(blackboard, f"Voice: {'ON' if is_voice_on else 'OFF'} (v)", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 127, 0))

        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("thresh", thresh)
        
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q') or keypress == ord('c'):
            break
        elif keypress == ord('v'):
            is_voice_on = not is_voice_on

    return 2 if keypress == ord('c') else 0 # 'c' for calculator mode

# --- 5. Main Execution ---

def main():
    """Main function to switch between modes."""
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("FATAL ERROR: Could not open webcam.")
        return

    # --- Warm-up prediction ---
    # This loads the model onto the GPU/CPU to prevent lag on the first frame.
    print("Warming up model...")
    keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
    print("Ready.")
    
    # Start in Text Mode (1)
    mode = 1
    while True:
        if mode == 1:
            print("Entering Text Mode. Press 'c' to switch to Calculator, 'q' to quit.")
            mode = text_mode(cam)
        elif mode == 2:
            print("Entering Calculator Mode. Press 't' to switch to Text, 'q' to quit.")
            mode = calculator_mode(cam)
        else:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()