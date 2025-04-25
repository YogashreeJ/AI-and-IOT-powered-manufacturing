from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, features)),
    LSTM(50, activation='relu'),
    Dense(1, activation='sigmoid')  # Predict failure (1/0)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
