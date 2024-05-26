# coding= UTF-8
import numpy as np
import pandas as pd
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fix random seed number
def train(language):
    np.random.seed(7)
    match language:
        case "chinese":
            file = "data/pickles/preprocessed_data_chinese.pkl"
        case "malay":
            file = "data/pickles/preprocessed_data_malay.pkl"
        case "tamil":
            file = "data/pickles/preprocessed_data_tamil.pkl"

    df = pd.read_pickle(file)
    df['mfcc'] = df['mfcc'].apply(lambda x: x.flatten())

    # Ensure mfcc is a consistent length
    mfcc_length = df['mfcc'].apply(len).max()
    df['mfcc'] = df['mfcc'].apply(lambda x: np.pad(x, (0, mfcc_length - len(x)), mode='constant'))

    # Convert mfcc column into multiple columns
    mfcc_features = np.stack(df['mfcc'].values)
    df_mfcc = pd.DataFrame(mfcc_features, index=df.index)

    # Combine all features
    X = pd.concat([df[['speech_rate', 'pause_rate', 'pronunciation_accuracy']], df_mfcc], axis=1)
    y = df['fluency'].astype(int)

    # Convert all column names to strings
    X.columns = X.columns.astype(str)

    # Print shapes for debugging
    print(X.shape)
    print(y.shape)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    number_of_features = X.shape[1]  # This is variable with each run
    number_of_classes = 4

    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train - 1, num_classes=number_of_classes)
    y_test = keras.utils.to_categorical(y_test - 1, num_classes=number_of_classes)

    # Neural Network Architecture
    model = Sequential()

    # 1st Layer with increased dropout rate
    model.add(Dense(512, input_dim=number_of_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))  # Increased dropout rate

    # 2nd Layer with increased dropout rate
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))  # Increased dropout rate

    # 3rd Layer. Output 4 neurons corresponding to the number of classes
    model.add(Dense(number_of_classes, activation='softmax'))


    # Model Compilation. Loss for multi-class classification problem
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),  # Adam optimizer
                  metrics=['accuracy'])

    # Early stopping and learning rate reduction on plateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)  # Increased patience
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.0001)  # Increased patience

    # Train and test
    model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr])
    score, acc = model.evaluate(X_test, y_test, batch_size=64)
    print("Test loss:", score)
    print("Test accuracy:", acc)
    
    # Save the model
    model.save("models/model_"+language+".keras")
    print("Model saved successfully.")

    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    # Save as pickle
    X_train_df.to_pickle('data/pickles/'+language+'_X_train.pkl')
    X_test_df.to_pickle('data/pickles/'+language+'_X_test.pkl')
    y_train_df.to_pickle('data/pickles/'+language+'_y_train.pkl')
    y_test_df.to_pickle('data/pickles/'+language+'_y_test.pkl')

if __name__ == "__main__":
    train("malay")
    train("chinese")
    train("tamil")
