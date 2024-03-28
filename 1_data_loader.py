import pandas as pd
from datasets import load_dataset
import os
import shutil

# Save train set
trainset = load_dataset("mispeech/speechocean762", split="train")

train_data = {
    'speaker': [sample['speaker'] for sample in trainset],
    'audio_path': [sample['audio']['path'] for sample in trainset],
    'accuracy': [sample['accuracy'] for sample in trainset],
    'fluency': [sample['fluency'] for sample in trainset],
    'prosodic': [sample['prosodic'] for sample in trainset],
    'age': [sample['age'] for sample in trainset],
    'sampling_rate': [sample['audio']['sampling_rate'] for sample in trainset],
    'text': [sample['text'] for sample in trainset]
}

train_df = pd.DataFrame(train_data)
train_df['audio_path'] = 'data/recordings/raw/WAVE/SPEAKER' + train_df['speaker'] + '/' + train_df['audio_path'].str.replace(r'\.wav$', '.WAV', case=False, regex=True)


# Define the path to the 'recordings' folder
recordings_path = 'data/recordings/raw/WAVE'
for speaker_folder in os.listdir(recordings_path):
    speaker_folder_path = os.path.join(recordings_path, speaker_folder)
    if os.path.isdir(speaker_folder_path):
        # Iterate through each file in the speaker's folder
        for filename in os.listdir(speaker_folder_path):
            if filename.endswith('.WAV'):
                # Construct the old and new file paths
                old_file_path = os.path.join(speaker_folder_path, filename)
                new_filename = f"{speaker_folder}-{filename}"
                new_file_path = os.path.join("data/recordings", new_filename)
                # Move and rename the file
                shutil.copy(old_file_path, new_file_path)
                print(f"Moved and renamed {old_file_path} to {new_file_path}")

                train_df.loc[train_df['audio_path'] == old_file_path, 'audio_path'] = new_file_path

train_df = train_df.drop(columns=['speaker'])

# Save test set
testset = load_dataset("mispeech/speechocean762", split="test")
test_data = {
    'speaker': [sample['speaker'] for sample in testset],
    'audio_path': [sample['audio']['path'] for sample in testset],
    'accuracy': [sample['accuracy'] for sample in testset],
    'fluency': [sample['fluency'] for sample in testset],
    'prosodic': [sample['prosodic'] for sample in testset],
    'age': [sample['age'] for sample in testset],
    'sampling_rate': [sample['audio']['sampling_rate'] for sample in testset],
    'text': [sample['text'] for sample in testset]
}

test_df = pd.DataFrame(test_data)
test_df['audio_path'] = 'data/recordings/SPEAKER' + test_df['speaker'] + '-' + test_df['audio_path'].str.replace(r'\.wav$', '.WAV', case=False, regex=True)
test_df = test_df.drop(columns=['speaker'])

train_df.to_pickle("data/pickles/train.pkl")
test_df.to_pickle("data/pickles/test.pkl")