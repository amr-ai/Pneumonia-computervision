import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('E:\codes\Python\PNEUMONIA\code\best_model.h5')

# GUI Application
# GUI Application
class PneumoniaDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumonia Detector - Maiden GUI")

        # Increase font size
        self.label_font = ("Helvetica", 16)

        self.label = tk.Label(root, text="Select a Test Image:", font=self.label_font)
        self.label.pack(pady=20)

        self.image_path = None
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=20)

        self.browse_button = tk.Button(root, text="Browse", command=self.browse_image, font=self.label_font)
        self.browse_button.pack(pady=20)

        self.detect_button = tk.Button(root, text="Detect Pneumonia", command=self.detect_pneumonia, font=self.label_font)
        self.detect_button.pack(pady=20)

    def browse_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.display_image()

    def display_image(self):
        img = Image.open(self.image_path)
        img = img.resize((200, 200), Image.ANTIALIAS)  # Increase the size of the displayed image
        img = ImageTk.PhotoImage(img)
        self.img_label.config(image=img)
        self.img_label.image = img

    def detect_pneumonia(self):
        if self.image_path:
            # Load and preprocess the image for prediction
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, (150, 150))
            img = np.expand_dims(img, axis=0) / 255.0

            # Make the prediction
            prediction = model.predict(img)
            class_idx = np.argmax(prediction)

            # Display the result
            if class_idx == 0:
                result = "PNEUMONIA"
            else:
                result = "NORMAL"

            self.label.config(text=f"Prediction: {result}")
        else:
            self.label.config(text="Please select a test image first.")

# Create the GUI
root = tk.Tk()
app = PneumoniaDetectorApp(root)
root.geometry("500x500")  # Set the initial window size
root.mainloop()
