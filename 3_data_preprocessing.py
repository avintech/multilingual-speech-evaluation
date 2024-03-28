import joblib
from joblib import dump, load
import modules.prepare_data as prepare_data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_pickle("data/pickles/augmented_data.pkl")

#Feature Extraction
try:
    for index, row in df.iterrows():
        try:
            data = prepare_data.load_audio(row['text'],row['audio_path'])
            #df.at[index,'wav_file_path'] = data['wav_file_path']
            df.at[index,'pronunciation_accuracy'] = data['pronunciation_accuracy']
            df.at[index,'speech_rate'] = data['speech_rate']
            df.at[index,'pause_rate'] = data['pause_rate']
            #df.at[index,'mfcc'] = data['mfcc']
            #df.at[index,'mean_pitch'] = data['mean_pitch']
            #df.at[index,'pitch_range'] = data['pitch_range']
            #df.at[index,'std_pitch'] = data['std_pitch']
        except Exception as ex:
            print(row['audio_path'])
            print(ex)
        finally:
            print(df.iloc[index])
            print("-" * 20)
except Exception as ex:
    print(ex)
finally:
    df.to_pickle("data/pickles/preprocessing_unmapped_data.pkl")

df = df.dropna(subset=['speech_rate','pause_rate'])

#Normalise data
columns_to_normalize = ['speech_rate','pause_rate','pronunciation_accuracy']
scaler = MinMaxScaler()
scaler.fit(df[columns_to_normalize])
df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
joblib.dump(scaler, 'models/scaler.joblib') #Save Scaler

#Map fluency values to based on rubrics provided
def map_fluency(value):
    if 0 <= value <= 3:
        return 0
    elif 4 <= value <= 5:
        return 1
    elif 6 <= value <= 7:
        return 2
    elif 8 <= value <= 10:
        return 3
    else:
        return None
df['fluency'] = df['fluency'].apply(map_fluency)

df.to_pickle("data/pickles/preprocessing_mapped_data.pkl")