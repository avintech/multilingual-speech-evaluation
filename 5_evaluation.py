from joblib import load
import os
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import modules.prepare_data as prepare_data

scaler = load('models/scaler.joblib')
xgboost_model = load('models/xgboost_model.joblib')
random_forest_model = load('models/random_forest_model.joblib')

file_path = "data/pickles/filled_evaluation_data.pkl"
if os.path.exists(file_path):
    print(file_path + " already exists.")
    new_data = pd.read_pickle(file_path)
else:
    new_data = pd.read_pickle("data/pickles/test.pkl")
    for index, row in new_data.iterrows():
        try:
            data = prepare_data.load_audio(row['text'],row['audio_path'])
            #new_data.at[index,'wav_file_path'] = data['wav_file_path']
            new_data.at[index,'pronunciation_accuracy'] = data['pronunciation_accuracy']
            new_data.at[index,'speech_rate'] = data['speech_rate']
            new_data.at[index,'pause_rate'] = data['pause_rate']
            #new_data.at[index,'mfcc'] = data['mfcc']
            #new_data.at[index,'mean_pitch'] = data['mean_pitch']
            #new_data.at[index,'pitch_range'] = data['pitch_range']
            #new_data.at[index,'std_pitch'] = data['std_pitch']
            new_data_norm = pd.DataFrame([data])
            new_data_norm = new_data_norm.drop(columns=['wav_file_path'])
            new_data_normalized = scaler.transform(new_data_norm)
            new_data.at[index,'gs_predicted_fluency'] = xgboost_model.predict(new_data_normalized)
            new_data.at[index,'rf_predicted_fluency'] = random_forest_model.predict(new_data_normalized)
        except Exception as ex:
            print(row['audio_path'])
            print(ex)
        finally:
            print(new_data.iloc[index])
            print("-" * 20)

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
        
    new_data['fluency'] = new_data['fluency'].apply(map_fluency)
    new_data.to_pickle(file_path)

gs_predicted_fluency = new_data['gs_predicted_fluency'].values
rf_predicted_fluency = new_data['rf_predicted_fluency'].values
actual_fluency = new_data['fluency'].values

accuracy_xg = accuracy_score(actual_fluency, gs_predicted_fluency)
accuracy_rf = accuracy_score(actual_fluency, rf_predicted_fluency) 
print(f"XGBoost Model Accuracy score: {accuracy_xg}")
print(f"Random Forest Accuracy score: {accuracy_rf}")

pearsonr_xg, _ = pearsonr(actual_fluency, gs_predicted_fluency)
pearsonr_rf, _ = pearsonr(actual_fluency, rf_predicted_fluency) 
print(f"XGBoost Model Pearson correlation coefficient: {pearsonr_xg}")
print(f"Random Forest Model Pearson correlation coefficient: {pearsonr_rf}")

precision_gs, recall_gs, f1_gs , _ = precision_recall_fscore_support(actual_fluency, gs_predicted_fluency)
precision_rf, recall_rf, f1_rf , _ = precision_recall_fscore_support(actual_fluency, rf_predicted_fluency)

print("XGBoost Class Performance:")
print(f"{'Class':<10}{'Precision':<10}{'Recall':<10}{'F1 Score':<10}")
for i, class_name in enumerate([0,1,2,3]):
    print(f"{class_name:<10}{precision_gs[i]:<10.2f}{recall_gs[i]:<10.2f}{f1_gs[i]:<10.2f}")


print("\nRandom Forest Class Performance:")
print(f"{'Class':<10}{'Precision':<10}{'Recall':<10}{'F1 Score':<10}")
for i, class_name in enumerate([0,1,2,3]):
    print(f"{class_name:<10}{precision_rf[i]:<10.2f}{recall_rf[i]:<10.2f}{f1_rf[i]:<10.2f}")