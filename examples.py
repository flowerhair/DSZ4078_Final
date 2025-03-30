# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 22:11:44 2025

@author: martin vojtíšek
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def show_shift(img_path, w_shift, h_shift):
    
    # Load an example image (640x480)
    img = load_img(img_path)  # Load image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Create ImageDataGenerator with shift
    datagen = ImageDataGenerator(width_shift_range=w_shift, height_shift_range=h_shift)
    
    # Generate augmented image
    augmented_img = next(datagen.flow(img_array, batch_size=1))[0].astype("uint8")
    
    # Plot Original vs Shifted Image
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[1].imshow(augmented_img)
    axes[1].set_title("Shifted Image")
    plt.show()