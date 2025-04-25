import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# Load dataset
df = pd.read_csv("D:\AI & IOT based MANUFACTURING INT\dataset.csv")

# Separate features and target variable
X = df.drop(columns=["Failure_Risk"])  # Features
y = df["Failure_Risk"]  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]  # Balance classes

# Train XGBoost model with class weighting
model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=5,
                          scale_pos_weight=scale_pos_weight)

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
