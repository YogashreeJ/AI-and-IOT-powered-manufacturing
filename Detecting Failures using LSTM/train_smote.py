from imblearn.over_sampling import SMOTE
import pandas as pd

# Load dataset
file_path = r"D:\AI & IOT based MANUFACTURING INT\dataset.csv"  # Change to your dataset path
df = pd.read_csv(file_path)

# Drop any remaining missing values
df.dropna(inplace=True)

# Features and target variable
X = df.drop(columns=["Failure_Risk"])
y = df["Failure_Risk"]

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a balanced dataset
balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df["Failure_Risk"] = y_resampled

# Save to CSV
balanced_file_path = "balanced_dataset.csv"  # Modify as needed
balanced_df.to_csv(balanced_file_path, index=False)

print(f"Balanced dataset saved as {balanced_file_path}")
