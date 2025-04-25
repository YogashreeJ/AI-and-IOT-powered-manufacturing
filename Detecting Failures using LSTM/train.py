import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv(r"D:\AI & IOT based MANUFACTURING INT\XGBoost\dataset.csv")  # Change to your dataset path

# Separate features and target
X = df.drop(columns=["Failure_Risk"])
y = df["Failure_Risk"]

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights for imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Train XGBoost with weighted class handling
xgb_weighted = xgb.XGBClassifier(eval_metric="logloss", n_estimators=50, max_depth=5,
                                 scale_pos_weight=scale_pos_weight)
xgb_weighted.fit(X_train, y_train)

# Make predictions
y_pred_weighted = xgb_weighted.predict(X_test)

# Evaluate model
weighted_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_weighted),
    "Precision": precision_score(y_test, y_pred_weighted),
    "Recall": recall_score(y_test, y_pred_weighted),
    "F1-Score": f1_score(y_test, y_pred_weighted)
}

print(weighted_metrics)
