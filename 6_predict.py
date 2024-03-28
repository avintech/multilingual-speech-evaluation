import argparse
import joblib
from joblib import load
import modules.prepare_data as prepare_data
import pandas as pd

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script to demonstrate adding args with argparse.')

    # Add arguments
    parser.add_argument('--model', type=str, help='Choose model: 0 for XGBoost, 1 for Random Forest', required=True)

    #provided text for new prediction audio
    text_file = "data/reading-passage.txt"
    with open(text_file, 'r') as file:
        provided_text = file.read()

    #prepare the new audio and extract features
    recording_file = "english1.mp3"
    data = prepare_data.load_audio(provided_text,recording_file)

    #use previous scaler to scale the new prediction to fit into the model
    data = pd.DataFrame([data])
    data = data.drop(columns=['wav_file_path','real_and_transcribed_words_ipa'])
    scaler = joblib.load('models/scaler.joblib')

    new_data_normalized = scaler.transform(data)

    # Load the model from the file
    # 0 for XGBoost, 1 for Random Forest
    args = parser.parse_args()
    print(args.model)
    if args.model == '0':
        loaded_model = load('models/xgboost_model.joblib')
    elif args.model == '1':
        loaded_model = load('models/random_forest_model.joblib')
    else:
        exit
    predictions = loaded_model.predict(new_data_normalized)
    print("Fluency Score: " + str(predictions[0]))

if __name__ == "__main__":
    main()