import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
import xgboost as xgb
from joblib import dump

def train(language):
    match language:
        case "chinese":
            file = "data/pickles/preprocessed_data_chinese.pkl"
        case "malay":
            file = "data/pickles/preprocessed_data_malay.pkl"
        case "tamil":
                file = "data/pickles/preprocessed_data_tamil.pkl"
    df = pd.read_pickle(file)

    #Load data
    X = df[['speech_rate','pause_rate','pronunciation_accuracy']]  # features
    y = df['fluency'].astype(int)  # target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #XGBoost
    try:
        xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=4, seed=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2, 0.0001],
            'max_depth': [4, 6, 8],
            'colsample_bytree': [0.3, 0.7, 1.0],
            'subsample': [0.6, 0.8, 1.0]
        }

        xg_grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
        xg_grid_search.fit(X_train, y_train)
        xg_grid_search_model = xg_grid_search.best_estimator_
        dump(xg_grid_search_model, 'models/xgboost_model_'+language+'.joblib')
        print("XGBoost Model Saved")
    except Exception as ex:
        print(ex)

    #Random Forest Classifier
    try:
        rf = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=0)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8]
        }

        rf_grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
        rf_grid_search.fit(X_train, y_train)
        rf_grid_search = rf_grid_search.best_estimator_
        dump(rf_grid_search, 'models/random_forest_model_'+language+'.joblib')
        print("Random Forest Model Saved")
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    train("malay")