from joblib import load
import os
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from modules.prepare_data import load_audio

from datasets import load_dataset


# Set the environment variable
os.environ['HF_TOKEN'] = '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'

# Now you can access it using os.environ or pass it to functions that need it
hf_token = os.environ['HF_TOKEN']

def eval(language):
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
                
    common_voice = load_dataset(dataset_path, split="test", use_auth_token=True)
    
    # Initialize DataFrame
    new_data = pd.DataFrame(common_voice)
    
    scaler = load('models/scaler_'+language+'.joblib')
    #xgboost_model = load('models/xgboost_model_'+language+'.joblib')
    random_forest_model = load('models/random_forest_model_'+language+'.joblib')

    file_path = "data/pickles/filled_evaluation_data.pkl"
    if os.path.exists(file_path):
        print(file_path + " already exists.")
        new_data = pd.read_pickle(file_path)
    else:
        for index, row in new_data.iterrows():
            try:
                data = load_audio(whisper_lang, new_data.at[index, 'original_script'], common_voice[index]['audio_path'])
                #new_data.at[index,'wav_file_path'] = data['wav_file_path']
                new_data.at[index,'pronunciation_accuracy'] = data['pronunciation_accuracy']
                new_data.at[index,'speech_rate'] = data['speech_rate']
                new_data.at[index,'pause_rate'] = data['pause_rate']
                new_data.at[index,'mfcc'] = data['mfcc']
                print(data)
                #new_data.at[index,'mean_pitch'] = data['mean_pitch']
                #new_data.at[index,'pitch_range'] = data['pitch_range']
                #new_data.at[index,'std_pitch'] = data['std_pitch']
                new_data_norm = pd.DataFrame([data])
                new_data_norm = new_data_norm.drop(columns=['real_and_transcribed_words','real_and_transcribed_words_ipa'])
                new_data_norm.reset_index(drop=True, inplace=True)
                new_data_normalized = scaler.transform(new_data_norm)
                #new_data.at[index,'gs_predicted_fluency'] = xgboost_model.predict(new_data_normalized)
                new_data.at[index,'rf_predicted_fluency'] = random_forest_model.predict(new_data_normalized)
                
            except Exception as ex:
                print(row['audio_path'])
                print(ex)
        new_data.to_pickle(file_path)

    rf_predicted_fluency = new_data['rf_predicted_fluency'].values
    actual_fluency = new_data['fluency'].values

    accuracy_rf = accuracy_score(actual_fluency, rf_predicted_fluency) 
    print(f"Random Forest Accuracy score: {accuracy_rf}")

    pearsonr_rf, _ = pearsonr(actual_fluency, rf_predicted_fluency) 
    print(f"Random Forest Model Pearson correlation coefficient: {pearsonr_rf}")

    precision_rf, recall_rf, f1_rf , _ = precision_recall_fscore_support(actual_fluency, rf_predicted_fluency)

    print("\nRandom Forest Class Performance:")
    print(f"{'Class':<10}{'Precision':<10}{'Recall':<10}{'F1 Score':<10}")
    for i, class_name in enumerate([0,1,2,3]):
        print(f"{class_name:<10}{precision_rf[i]:<10.2f}{recall_rf[i]:<10.2f}{f1_rf[i]:<10.2f}")



if __name__ == "__main__":
    eval("malay")
