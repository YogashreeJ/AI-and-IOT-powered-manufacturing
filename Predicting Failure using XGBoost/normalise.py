import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
X = pd.read_csv(r"D:\AI & IOT based MANUFACTURING INT\LSTM\X_features.csv")

# Identify the timestamp column (assuming it's the first column)
timestamp_col = X.columns[3]

# Separate the timestamp and numeric data
timestamps = X[timestamp_col]
X_numeric = X.drop(columns=[timestamp_col])  # Drop timestamp for scaling

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numeric features
X_scaled = scaler.fit_transform(X_numeric)

# Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)

# Add back the timestamp column
X_scaled_df.insert(3, timestamp_col, timestamps)

# Save the normalized dataset
X_scaled_df.to_csv("X_features_normalized.csv", index=False)

print("Normalization complete. Saved as X_features_normalized.csv.")
