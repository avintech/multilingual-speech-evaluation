import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from joblib import dump
import numpy as np
from sklearn.metrics import classification_report

def train(language):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    X_train = pd.read_pickle('data/pickles/'+language+'_X_train.pkl')
    X_test = pd.read_pickle('data/pickles/'+language+'_X_test.pkl')
    y_train = pd.read_pickle('data/pickles/'+language+'_y_train.pkl')
    y_test = pd.read_pickle('data/pickles/'+language+'_y_test.pkl')

    # #XGBoost
    # try:
    #     xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=4, seed=42)
    #     param_grid = {
    #         'n_estimators': [100, 200, 300],
    #         'learning_rate': [0.01, 0.1, 0.2, 0.0001],
    #         'max_depth': [4, 6, 8],
    #         'colsample_bytree': [0.3, 0.7, 1.0],
    #         'subsample': [0.6, 0.8, 1.0]
    #     }

    #     xg_grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    #     xg_grid_search.fit(X_train, y_train)
    #     xg_grid_search_model = xg_grid_search.best_estimator_
    #     dump(xg_grid_search_model, 'models/xgboost_model_'+language+'.joblib')
    #     print("XGBoost Model Saved")
    # except Exception as ex:
    #     print(ex)

    #Random Forest Classifier
    try:
        rf = RandomForestClassifier( random_state=0)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8]
        }

        rf_grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
        rf_grid_search.fit(X_train, y_train)
        rf_grid_search = rf_grid_search.best_estimator_
        dump(rf_grid_search, 'models/random_forest_model_'+language+'.joblib')
        print("Random Forest Model "+language+" Saved")

        # Evaluate precision, recall, and f1-score
        y_pred = rf_grid_search.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        report = classification_report(y_test_classes, y_pred_classes)
        print(report)
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    train("malay") #doesnt work
    train("chinese")
    train("tamil")