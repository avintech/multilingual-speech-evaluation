import modules.audio_augmentation as audio_augmentation
import os
import pandas as pd
import shutil

#save augmented data
df = pd.read_pickle("data/pickles/train.pkl")

folder_path = "../recordings/augmented"
if not os.path.exists(folder_path):
    # If the folder does not exist, create it
    os.makedirs(folder_path)
    result = "Folder created."
else:
    result = "Folder already exists."
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)

for index, row in df.iterrows():
    new_index = len(df)
    augmented_path = audio_augmentation.augment_audio(row['audio_path'])
    new_row_data = {'audio_path': augmented_path, 'accuracy': row['accuracy'],'fluency': row['fluency'],'prosodic': row['prosodic'],
                    'age': row['age'],'sampling_rate': row['sampling_rate'],'text': row['text']}
    df.loc[new_index] = new_row_data
    
df.to_pickle("data/pickles/augmented_data.pkl")