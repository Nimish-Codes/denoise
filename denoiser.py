import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import MeanSquaredError

# Define the U-Net-like autoencoder model for denoising color images
def denoising_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    # Decoder
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    up1 = UpSampling2D((2, 2))(conv2)
    
    # Output layer
    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up1)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
model = denoising_model(input_shape=(256, 256, 3))

# Compile the model with perceptual loss
model.compile(optimizer='adam', loss='mse')  # You can also try MeanSquaredError()

# Streamlit app
st.title("Image Denoiser")

# File uploader widget for uploading a noisy image
noisy_image = st.file_uploader("Upload a Noisy Image", type=["jpg", "jpeg", "png"])

if noisy_image is not None:
    # Load and preprocess the noisy image
    noisy_img = image.load_img(noisy_image, target_size=(256, 256))
    noisy_img = image.img_to_array(noisy_img) / 255.0
    noisy_img = np.expand_dims(noisy_img, axis=0)

    # Print some information about the input image
    st.image(noisy_img, caption="Noisy Image", use_column_width=True)
    st.write("Noisy Image Shape:", noisy_img.shape)

    # Train the model (you may need to adjust the training process based on your needs)
    model.fit(noisy_img, noisy_img, epochs=100, batch_size=1)

    # Denoise the input noisy image
    denoised_img = model.predict(noisy_img)

    # Print some information about the denoised image
    st.image(denoised_img, caption="Denoised Image", use_column_width=True)
    st.write("Denoised Image Shape:", denoised_img.shape)
