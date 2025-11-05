import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pyttsx3
from threading import Thread
from glob import glob
import pyautogui

# --- GUI Libraries ---
import customtkinter as ctk
from PIL import Image, ImageTk

# --- 1. Application Setup ---

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Configure GUI appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- 2. Load Models & Assets ---

print("Loading deep learning model...")
model = load_model('cnn_model_keras.h5')

# --- 3. Core Helper Functions ---

def load_label_map():
    """
    Loads the "dictionary" (label map) that maps numbers back to names.
    e.g., {0: 'app_switch', 1: 'close_window', ...}
    """
    if not os.path.exists("label_map.pkl"):
        print("FATAL ERROR: 'label_map.pkl' file not found.")
        print("Please run 'load_images.py' script first.")
        exit()
        
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    
    # Create the integer-to-string inverse map
    inverse_label_map = {v: k for k, v in label_map.items()}
    return inverse_label_map

def get_image_size():
    """Finds one image to see what size (e.g., 50x50) our model expects."""
    all_images = glob("gestures/*/*.jpg")
    if not all_images:
        print("FATAL ERROR: No images found in 'gestures' folders.")
        print("Please run 'create_gestures.py' first.")
        exit()
        
    img = cv2.imread(all_images[0], 0)
    if img is None:
        print(f"FATAL ERROR: Could not read sample image {all_images[0]}")
        exit()
    # Return (height, width)
    return img.shape[0], img.shape[1]

def keras_process_image(img):
    """
    This is the "translator" that formats the hand image
    to be the *exact* 50x50, normalized, 4D format the model expects.
    """
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0 # Normalize (0.0 to 1.0)
    img = np.reshape(img, (1, image_y, image_x, 1)) # (1, 50, 50, 1)
    return img

def keras_predict(model, image):
    """Asks the "brain" what it sees."""
    processed = keras_process_image(image)
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = np.argmax(pred_probab) # Find the highest probability
    return max(pred_probab), pred_class

def get_pred_from_contour(contour, thresh):
    """Crops the hand from the mask and sends it to the model."""
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1 + h1, x1:x1 + w1]
    
    # Pad the image to make it square
    if w1 > h1:
        padding = int((w1 - h1) / 2)
        save_img = cv2.copyMakeBorder(save_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        padding = int((h1 - w1) / 2)
        save_img = cv2.copyMakeBorder(save_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, (0, 0, 0))
    
    pred_probab, pred_class = keras_predict(model, save_img)
    
    # Only accept the prediction if the model is > 70% confident
    if pred_probab * 100 > 70:
        text = inverse_label_map[pred_class]
        return text
    
    return ""

def get_img_contour_thresh(img):
    """
    This is the "skin detector." It finds the hand in the frame
    using YCrCb color space, which is great for bright rooms.
    """
    x, y, w, h = 300, 100, 300, 300
    img = cv2.flip(img, 1) # Mirror mode
    
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin_cr_min = 133
    skin_cr_max = 173
    skin_cb_min = 77
    skin_cb_max = 127
    
    # Create the skin mask
    skin_mask = cv2.inRange(
        img_ycrcb, 
        np.array([0, skin_cr_min, skin_cb_min]), 
        np.array([255, skin_cr_max, skin_cb_max])
    )
    
    # Clean the mask
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    skin_mask = cv2.filter2D(skin_mask, -1, disc)
    blur = cv2.GaussianBlur(skin_mask, (11, 11), 0)
    thresh = cv2.medianBlur(blur, 15)
    
    # Crop the mask to the green box
    thresh_roi = thresh[y:y + h, x:x + w]
    
    # Find the hand's outline
    contours = cv2.findContours(thresh_roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    
    # Draw the green box on the color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img, contours, thresh_roi # Return the cropped B&W mask

def say_text(text):
    """Runs the text-to-speech in a separate thread so it doesn't freeze the app."""
    global is_voice_on, engine
    if not is_voice_on or engine is None:
        return
    try:
        while engine._inLoop: # Wait for the last speech to finish
            pass
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"pyttsx3 error: {e}")

# --- Load global variables ---
inverse_label_map = load_label_map()
image_y, image_x = get_image_size()

# --- THIS IS THE TEXT-TO-SPEECH FIX ---
# We wrap the init() in a try-except. If it fails (e.g., no audio device),
# 'engine' will be 'None' and the app won't crash on exit.
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
except Exception as e:
    print(f"Warning: pyttsx3 init failed. Voice output will be disabled. Error: {e}")
    engine = None
# --- END OF FIX ---

is_voice_on = True

# --- 4. The Main Application GUI Class ---

class App(ctk.CTk):
    """This class is our entire application window."""
    
    def __init__(self):
        super().__init__()
        
        # --- Application Window Setup ---
        self.title("Gesture-Based Productivity Suite")
        self.geometry("1100x600")
        
        # --- App's "memory" ---
        self.text = ""
        self.command = ""
        self.count_same_frame = 0
        self.old_text = ""
        
        # --- Camera Setup ---
        self.cam = cv2.VideoCapture(1)
        if not self.cam.isOpened():
            self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            print("FATAL ERROR: Cannot open webcam.")
            self.destroy()
            return
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # --- GUI Layout ---
        self.grid_columnconfigure(0, weight=3) # Video panel
        self.grid_columnconfigure(1, weight=1) # Status panel
        self.grid_rowconfigure(0, weight=1)
        
        # 1. Main Video Panel
        self.video_label = ctk.CTkLabel(self, text="Loading...")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # 2. Status & Diagnostic Panel (Right side)
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.status_frame.grid_rowconfigure(6, weight=1) # Spacer
        
        self.status_title = ctk.CTkLabel(self.status_frame, text="STATUS", font=ctk.CTkFont(size=20, weight="bold"))
        self.status_title.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.pred_label = ctk.CTkLabel(self.status_frame, text="Predicted: --", font=ctk.CTkFont(size=16))
        self.pred_label.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        
        self.command_label = ctk.CTkLabel(self.status_frame, text="COMMAND: --", font=ctk.CTkFont(size=16))
        self.command_label.grid(row=2, column=0, padx=20, pady=10, sticky="w")
        
        self.voice_button = ctk.CTkButton(self.status_frame, text="Voice: ON", command=self.toggle_voice)
        self.voice_button.grid(row=3, column=0, padx=20, pady=20)
        
        # 3. Diagnostic "Thresh" View
        self.thresh_title = ctk.CTkLabel(self.status_frame, text="Diagnostic View (What the AI sees)", font=ctk.CTkFont(size=16, weight="bold"))
        self.thresh_title.grid(row=4, column=0, padx=20, pady=(20, 10))
        
        self.thresh_label = ctk.CTkLabel(self.status_frame, text="")
        self.thresh_label.grid(row=5, column=0, padx=20, pady=10)
        
        # --- Model Warm-up and Main Loop Start ---
        print("Warming up model...")
        keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
        print("Ready.")
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle "X" button
        self.update_frame() # Start the app's "heartbeat"

    def toggle_voice(self):
        """Toggles the text-to-speech feature on and off."""
        global is_voice_on
        is_voice_on = not is_voice_on
        if is_voice_on:
            self.voice_button.configure(text="Voice: ON")
        else:
            self.voice_button.configure(text="Voice: OFF")

    def update_frame(self):
        """
        This is the "heartbeat" of our app. It runs over and over.
        """
        
        # 1. Grab camera frame
        ret, img = self.cam.read()
        if not ret:
            print("Error: Failed to grab frame.")
            self.after(15, self.update_frame)
            return
            
        # 2. Process the image (Skin detection)
        img, contours, thresh_roi = get_img_contour_thresh(img)
        
        # 3. Run prediction logic
        self.old_text = self.text
        
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                self.text = get_pred_from_contour(contour, thresh_roi)
                
                if self.old_text == self.text and self.text != "":
                    self.count_same_frame += 1
                else:
                    self.count_same_frame = 0

                # 4. Run command logic (if gesture held for 20+ frames)
                if self.count_same_frame > 20:
                    
                    if self.text == "nothing":
                        self.command = ""
                    
                    elif self.text == "app_switch":
                        pyautogui.hotkey('alt', 'tab')
                        self.command = "App Switch"
                        Thread(target=say_text, args=(self.command, )).start()
                        
                    elif self.text == "close_window":
                        pyautogui.hotkey('alt', 'f4')
                        self.command = "Close Window"
                        Thread(target=say_text, args=(self.command, )).start()
                        
                    elif self.text == "scroll_up":
                        pyautogui.scroll(200) # Positive scrolls up
                        self.command = "Scroll Up"
                    
                    elif self.text == "scroll_down":
                        pyautogui.scroll(-200) # Negative scrolls down
                        self.command = "Scroll Down"
                    
                    elif self.text == "volume_up":
                        pyautogui.press('volumeup')
                        self.command = "Volume Up"
                        Thread(target=say_text, args=(self.command, )).start()
                    
                    elif self.text == "volume_down":
                        pyautogui.press('volumedown')
                        self.command = "Volume Down"
                        Thread(target=say_text, args=(self.command, )).start()
                        
                    elif self.text == "screenshot":
                        pyautogui.press('printscreen')
                        self.command = "Screenshot"
                        Thread(target=say_text, args=("Screenshot taken", )).start()
                        
                    elif self.text == "play_pause":
                        pyautogui.press('space')
                        self.command = "Play / Pause"
                        Thread(target=say_text, args=(self.command, )).start()

                    self.count_same_frame = 0 # Reset counter
            
        else:
            self.text = "" # Clear prediction if no hand
            self.command = ""
        
        # 5. Update the GUI labels
        self.pred_label.configure(text=f"Predicted: {self.text}")
        self.command_label.configure(text=f"COMMAND: {self.command}")
        
        # 6. Convert Main Video Frame for GUI
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        ctk_image = ctk.CTkImage(light_image=pil_image, size=(640, 480))
        
        # 7. Convert Thresh Frame for GUI
        thresh_bgr = cv2.cvtColor(thresh_roi, cv2.COLOR_GRAY2BGR)
        thresh_rgb = cv2.cvtColor(thresh_bgr, cv2.COLOR_BGR2RGB)
        pil_image_thresh = Image.fromarray(thresh_rgb)
        # Make it smaller to fit on the side panel
        ctk_image_thresh = ctk.CTkImage(light_image=pil_image_thresh, size=(240, 240))
        
        # 8. Update the image widgets
        self.video_label.configure(image=ctk_image)
        self.video_label.image = ctk_image # Prevent garbage collection
        
        self.thresh_label.configure(image=ctk_image_thresh)
        self.thresh_label.image = ctk_image_thresh # Prevent garbage collection
        
        # 9. Repeat the loop every 15ms
        self.after(15, self.update_frame)

    def on_closing(self):
        """Called when the 'X' button is pressed."""
        print("Exiting application...")
        # Safely release the camera
        self.cam.release()
        # Destroy the GUI
        self.destroy()
        # Properly stop the TTS engine if it's running
        if engine:
            engine.stop()

# --- 5. Main Execution ---
if __name__ == "__main__":
    app = App() # Create the application
    app.mainloop() # Run the application