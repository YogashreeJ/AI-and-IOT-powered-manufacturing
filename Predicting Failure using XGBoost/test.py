import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

#  Load Data
X = np.load(r"D:\AI & IOT based MANUFACTURING INT\LSTM\X_lstm_sequences.npy")
y = np.load(r"D:\AI & IOT based MANUFACTURING INT\LSTM\y_lstm_sequences.npy")

#  Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Define Optimized LSTM Model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    LSTM(256, return_sequences=True, activation="tanh"),
    Dropout(0.2),

    LSTM(128, return_sequences=True, activation="tanh"),
    Dropout(0.2),

    LSTM(64, return_sequences=False, activation="tanh"),
    Dropout(0.1),

    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification
])

#  Compile the Model
optimizer = Adam(learning_rate=0.0008)  #  Slightly higher LR
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

#  Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the Model
history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_reduction]
)

#  Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
