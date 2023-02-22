import os
import IPython.display as ipd
import librosa
import librosa.display as lbd
import pandas as pd
from scipy.io import wavfile as wav
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 
import resampy
import llvmlite
import streamlit as st
from keras.models import load_model

# Load the model
model = load_model('saved_models/audio_classification.hdf5')
labelencoder = LabelEncoder()
labelencoder.classes_ = np.load('classes1.npy', allow_pickle=True)

def get_classification(audio_file):
    audio, sample_rate = librosa.load(audio_file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict(mfccs_scaled_features)
    classes = np.argmax(predicted_label,axis=1)
    prediction_class = labelencoder.inverse_transform(classes)[0] 
    return prediction_class

# Streamlit app code
st.title('Audio Classification Demo')

# Add an upload file widget
uploaded_file = st.file_uploader('Upload an audio file', type=['wav'])

# Display the uploaded audio file and make a prediction
if uploaded_file is not None:
    st.audio(uploaded_file)
    predicted_class = get_classification(uploaded_file)
    st.write(f'Predicted class: {predicted_class}')