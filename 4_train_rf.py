import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from joblib import dump
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

def train(language):
    X_train = pd.read_pickle(f'data/pickles/{language}_X_train.pkl')
    X_test = pd.read_pickle(f'data/pickles/{language}_X_test.pkl')
    y_train = pd.read_pickle(f'data/pickles/{language}_y_train.pkl')
    y_test = pd.read_pickle(f'data/pickles/{language}_y_test.pkl')

    # Convert y_train and y_test to 1-dimensional arrays if they are one-hot encoded
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.idxmax(axis=1).values
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.idxmax(axis=1).values

    # Resample the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': [None, 'sqrt', 'log2']
    }

    # Perform grid search with stratified cross-validation
    skf = StratifiedKFold(n_splits=5)
    rf_grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced'), 
                                  param_grid, cv=skf, scoring='accuracy')
    
    rf_grid_search.fit(X_train_res, y_train_res)
    best_rf = rf_grid_search.best_estimator_
    dump(best_rf, f'models/random_forest_model_{language}.joblib')
    print(f"Random Forest Model {language} Saved")

    # Evaluate precision, recall, and f1-score
    y_pred = best_rf.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    print(report)

    # Calculate and print ROC-AUC score if there are only two classes
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred)
        print("ROC-AUC Score:", roc_auc)

if __name__ == "__main__":
    train("tamil")


# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from imblearn.over_sampling import SMOTE
# from joblib import dump
# from sklearn.metrics import classification_report, roc_auc_score
# import numpy as np

# def train(language):
#     X_train = pd.read_pickle(f'data/pickles/{language}_X_train.pkl')
#     X_test = pd.read_pickle(f'data/pickles/{language}_X_test.pkl')
#     y_train = pd.read_pickle(f'data/pickles/{language}_y_train.pkl')
#     y_test = pd.read_pickle(f'data/pickles/{language}_y_test.pkl')

#     # Convert y_train and y_test to 1-dimensional arrays if they are one-hot encoded
#     if isinstance(y_train, pd.DataFrame):
#         y_train = y_train.idxmax(axis=1).values
#     if isinstance(y_test, pd.DataFrame):
#         y_test = y_test.idxmax(axis=1).values

#     # Resample the training data
#     smote = SMOTE(random_state=42)
#     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#     }

#     # Perform grid search with stratified cross-validation
#     skf = StratifiedKFold(n_splits=5)
#     rf_grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced'), 
#                                   param_grid, cv=skf, scoring='accuracy')
    
#     rf_grid_search.fit(X_train_res, y_train_res)
#     best_rf = rf_grid_search.best_estimator_
#     dump(best_rf, f'models/random_forest_model_{language}.joblib')
#     print(f"Random Forest Model {language} Saved")

#     # Evaluate precision, recall, and f1-score
#     y_pred = best_rf.predict(X_test)
    
#     report = classification_report(y_test, y_pred)
#     print(report)

#     # Calculate and print ROC-AUC score if there are only two classes
#     if len(np.unique(y_test)) == 2:
#         roc_auc = roc_auc_score(y_test, y_pred)
#         print("ROC-AUC Score:", roc_auc)

# if __name__ == "__main__":
#     train("tamil")
