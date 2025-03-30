# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 11:02:30 2025

@author: marti
"""
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json

# Load the model
input_model_my = tf.keras.models.load_model("./my_model_V2.keras")
input_model_tl = tf.keras.models.load_model("./tl_model_V2S.keras")
input_model_tl2 = tf.keras.models.load_model("./tl_model_V2M.keras")

with open("./class_mapping.json", "r") as f:
    class_dict = json.load(f)

selected_image_path = None

# Image preprocessing function (adjust based on your model's input shape)
#Model expects 224*224, normalized 0-1 images with rgb channels
def preprocess_image(image_path, target_size=(224, 224), scale_factor=255.0):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / scale_factor  # Normalize if required
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to select an image
def select_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        selected_image_path = file_path
        # Display image in UI
        img = Image.open(file_path)
        img = img.resize((224, 224))  # Resize for display
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text="")  # Clear previous label

# Function to classify the image using both models
def classify_image():
    if not selected_image_path:
        result_label.config(text="No image selected!")
        return
    
    img_array = preprocess_image(selected_image_path)
    img_array_tl = preprocess_image(selected_image_path, scale_factor=1.0)
    

    # Get predictions from both models
    pred1 = input_model_my.predict(img_array)
    pred2 = input_model_tl.predict(img_array_tl)
    pred3 = input_model_tl2.predict(img_array_tl)
    
    pc1 = np.argmax(pred1)
    pc2 = np.argmax(pred2)
    pc3 = np.argmax(pred3)
    ps1 = np.max(pred1)
    ps2 = np.max(pred2)
    ps3 = np.max(pred3)
    

    # Average the softmax outputs
    avg_pred = (pred1 + pred2 + pred3) / 3

    # Get the class with the highest probability
    predicted_class = np.argmax(avg_pred)
    confidence_score = np.max(avg_pred)  # Highest softmax probability

    # Display result with confidence score
    result_label.config(
        text=f"""
        MYMOD: {class_dict[str(pc1)]} ({ps1:.2f})
        ENV2S: {class_dict[str(pc2)]} ({ps2:.2f})
        ENV2M: {class_dict[str(pc3)]} ({ps3:.2f})
        AVG: {class_dict[str(predicted_class)]} ({confidence_score:.2f})
        """)

# GUI setup
root = tk.Tk()
root.title("Image Classifier")
root.geometry("360x380")

btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack()

image_label = tk.Label(root)
image_label.pack()

btn_label = tk.Button(root, text="Label Image", command=classify_image)
btn_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack()

root.mainloop()
