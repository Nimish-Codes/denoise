import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose
from tensorflow.keras.preprocessing import image

# Get the file path of the noisy image from the user
noisy_image_path = st.text_input("Enter the path of the noisy image:")

# Load and preprocess the noisy image
noisy_img = image.load_img(noisy_image_path)
noisy_img = image.img_to_array(noisy_img) / 255.0
noisy_img = np.expand_dims(noisy_img, axis=0)

# Define the autoencoder model for denoising color images
model = Sequential()

# Encoder
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(None, None, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

# Decoder
model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Use Mean Squared Error for color images

# Print a summary of the model architecture
st.text("Model Summary:")
model.summary()

# Denoise the input noisy image
denoised_img = model.predict(noisy_img)

# Plot original noisy and denoised images using Matplotlib and Streamlit
st.image(np.squeeze(noisy_img), caption='Noisy Image', use_column_width=True)
st.image(np.squeeze(denoised_img), caption='Denoised Image', use_column_width=True)
