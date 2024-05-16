import sys, os
import joblib
from modules.prepare_data import load_audio
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset, DatasetDict, concatenate_datasets

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
                
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(dataset_path, split="train", use_auth_token=True)
        common_voice["test"] = load_dataset(dataset_path, split="test", use_auth_token=True)
        common_voice_combine = concatenate_datasets([common_voice['train'], common_voice['test']])
        df = pd.DataFrame(common_voice_combine)
        #Do for training data
        for index, row in df.iterrows():
            try:
            
                print("processing audio......")
                data = load_audio(whisper_lang,df.at[index, 'original_script'],common_voice_combine[index]['audio_path'])
                print("processing audio completed!")
                df.at[index,'pronunciation_accuracy'] = data['pronunciation_accuracy']
                df.at[index,'speech_rate'] = data['speech_rate']
                df.at[index,'pause_rate'] = data['pause_rate']
                #df.at[index,'mfcc'] = data['mfcc']
                #df.at[index,'mean_pitch'] = data['mean_pitch']
                #df.at[index,'pitch_range'] = data['pitch_range']
                #df.at[index,'std_pitch'] = data['std_pitch']
            
            except Exception as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
            finally:
                print(df.iloc[index])
                print("-" * 20)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
    df = df.dropna(subset=['speech_rate','pause_rate'])

    #Normalise data
    columns_to_normalize = ['speech_rate','pause_rate','pronunciation_accuracy']
    scaler = MinMaxScaler()
    scaler.fit(df[columns_to_normalize])
    df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
    joblib.dump(scaler, 'models/scaler_'+language+'.joblib') #Save Scaler
    df.to_pickle("data/pickles/preprocessed_data_"+language+".pkl")

if __name__ == "__main__":
    pre_process("malay")