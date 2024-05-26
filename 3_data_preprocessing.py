import os
import joblib
from modules.prepare_data import load_audio
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset,concatenate_datasets

# Set the environment variable
os.environ['HF_TOKEN'] = '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'

# Now you can access it using os.environ or pass it to functions that need it
hf_token = os.environ['HF_TOKEN']

def pre_process(language):
    try:
        match language:
            case "chinese":
                dataset_path = "avintech/chinese_children_speech"
                whisper_lang = "zh"
            case "malay":
                dataset_path = "avintech/malay_batch1"
                whisper_lang = "ms"
            case "tamil":
                dataset_path = "avintech/tamil_children_speech"
                whisper_lang = "ta"
                
        common_voice = load_dataset(dataset_path, split="train", use_auth_token=True)
        common_voice2 = load_dataset(dataset_path, split="test", use_auth_token=True)
        common_voice = concatenate_datasets([common_voice,common_voice2])
        
        # Initialize DataFrame
        df = pd.DataFrame(common_voice)

        # Remove unnecessary initialization of 'mfcc' column
        df['mfcc'] = None
        df = df.astype({'mfcc': 'object'})

        # Process each row in the DataFrame
        for index, row in df.iterrows():
            try:
                print(f"Processing audio at index {index}...")
                data = load_audio(whisper_lang, df.at[index, 'original_script'], common_voice[index]['audio_path'])
                print(f"Audio processing completed for index {index}!")
                
                # Check the type and length of 'mfcc' data
                print(type(data['mfcc']))
                print(len(data['mfcc']))

                # Assign features to DataFrame
                df.at[index, 'pronunciation_accuracy'] = data['pronunciation_accuracy']
                df.at[index, 'speech_rate'] = data['speech_rate']
                df.at[index, 'pause_rate'] = data['pause_rate']
                df.at[index, 'mfcc'] = data['mfcc']
                # df.at[index, 'mean_pitch'] = data['mean_pitch']
                # df.at[index, 'pitch_range'] = data['pitch_range']
                # df.at[index, 'std_pitch'] = data['std_pitch']
            
            except Exception as ex:
                print(f"Error processing audio at index {index}: {ex}")
            finally:
                # Print the row for debugging purposes
                print(df.iloc[index])
                print("-" * 20)


    except Exception as e:
        print(f"Error in pre_process function: {e}")
        
    required_columns = ['speech_rate', 'pause_rate']
    if all(column in df.columns for column in required_columns):
        df = df.dropna(subset=required_columns)
        print("Rows with NaN values in 'speech_rate' or 'pause_rate' have been dropped.")
    else:
        print("One or both required columns are missing. No rows were dropped.")
    
    # Normalize data
    columns_to_normalize = ['speech_rate', 'pause_rate', 'pronunciation_accuracy']
    scaler = MinMaxScaler()
    scaler.fit(df[columns_to_normalize])
    df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
    joblib.dump(scaler, f'models/scaler_{language}.joblib')  # Save Scaler
    df.to_pickle(f"data/pickles/preprocessed_data_{language}.pkl")


if __name__ == "__main__":
    pre_process("tamil")
