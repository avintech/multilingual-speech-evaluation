#import audio_augmentation
from datasets import load_dataset
import joblib
from joblib import dump, load
import os
import pandas as pd
import prepare_data
from scipy.stats import pearsonr
import shutil
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text
import torch
import xgboost as xgb

# Page configuration
st.set_page_config(page_title="Speech Evaluation")

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Reading", "Picture Discussion"])

# Page: Image
if selection == "Picture Discussion":
    st.title("Picture Discussion Task")
    st.write("Please describe the below image to the best of your ability.")
    with st.container():
        st.image("images/image1.png")

# Page: Text
elif selection == "Reading":
    st.title("Reading Task")
    st.write("Please read the below text to the best of your ability.")
    provided_text = "Opera refers to a dramatic art form, originating in Europe, in which the emotional content is conveyed to the audience as much through music, both vocal and instrumental, as it is through the lyrics. By contrast, in musical theater an actor's dramatic performance is primary, and the music plays a lesser role. The drama in opera is presented using the primary elements of theater such as scenery, costumes, and acting. However, the words of the opera, or libretto, are sung rather than spoken. The singers are accompanied by a musical ensemble ranging from a small instrumental ensemble to a full symphonic orchestra."
    with st.container():
        st.write(provided_text)
     
def callback():
    if st.session_state.my_recorder_output:
        audio_bytes=st.session_state.my_recorder_output['bytes']
        with open('myfile2.wav', mode='bx') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes)
        data = prepare_data.load_audio(provided_text,'myfile2.wav')
        
        #use previous scaler to scale the new prediction to fit into the model
        data = pd.DataFrame([data])
        data = data.drop(columns=['wav_file_path'])
        scaler = joblib.load('my_scaler.joblib')
        new_data_normalized = scaler.transform(data)

        # Load the model from the file
        loaded_model = load('grid_search_model.joblib')
        # Use the loaded model
        predictions = loaded_model.predict(new_data_normalized) # Assuming you have an X_test set
        
        print(predictions)
mic_recorder(key='my_recorder', callback=callback)
