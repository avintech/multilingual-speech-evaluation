import argparse
import joblib
from joblib import load
import modules.prepare_data as prepare_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
import librosa


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script to demonstrate adding args with argparse.')

    # Add arguments
    parser.add_argument('--model', type=str, help='Choose model: 0 for MLP, 1 for Random Forest', required=True)
    parser.add_argument('--lang', type=str, help='Choose 1: malay, chinese, tamil', required=True)
    args = parser.parse_args()
    text_file = "data/reading-passage.txt"
    recording_file = "data/recordings/chinese/chinese_b2/0dc73844-8d4f-2b00-75f6-c6bc3d267377Text_002_Line_1.wav"
    
    predict(args.model,args.lang,text_file,recording_file)
    #provided text for new prediction audio

def predict(model,lang,text_file,recording_file):
    with open(text_file, 'r') as file:
        provided_text = file.read()

    print(provided_text)

    #prepare the new audio and extract features
    audio_array, sampling_rate = librosa.load(recording_file, sr=None)
    audio_data = {'array': audio_array, 'sampling_rate': sampling_rate}

    print(audio_data['array'])
    print(audio_data['sampling_rate'])

    data = prepare_data.load_audio(lang,provided_text,audio_data)

    #use previous scaler to scale the new prediction to fit into the model
    data = pd.DataFrame([data])
    data['mfcc'] = data['mfcc'].apply(lambda x: x.flatten())
    mfcc_length = data['mfcc'].apply(len).max()
    data['mfcc'] = data['mfcc'].apply(lambda x: np.pad(x, (0, mfcc_length - len(x)), mode='constant'))

    # Convert mfcc column into multiple columns
    mfcc_features = np.stack(data['mfcc'].values)
    df_mfcc = pd.DataFrame(mfcc_features, index=data.index)
    X = pd.concat([data[['speech_rate', 'pause_rate', 'pronunciation_accuracy']], df_mfcc], axis=1)
    X.columns = X.columns.astype(str)

    #Load scalar
    scaler = StandardScaler()
    X_train = pd.read_pickle("data/pickles/"+lang+"_X_train.pkl")
    scaler.fit(X_train)

    #Normalise new data
    new_data_scaled = scaler.transform(X)

    # Load the model from the file
    # 0 for XGBoost, 1 for Random Forest
    print(model)
    if model == '0':
        loaded_model = keras.models.load_model('models/model_'+lang+'.keras')
    elif model == '1':
        loaded_model = load('models/random_forest_model.joblib')
    else:
        exit

    y_pred = loaded_model.predict(new_data_scaled)
    y_pred_class = np.argmax(y_pred, axis=1)

    print("Fluency Score: " + str(y_pred_class))

if __name__ == "__main__":
    main()