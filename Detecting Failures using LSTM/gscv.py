import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# Load dataset
df = pd.read_csv(r"D:\AI & IOT based MANUFACTURING INT\dataset.csv")

# Drop missing values
df = df.dropna()

# Define features and target
X = df.drop(columns=["Failure_Risk"])
y = df["Failure_Risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Adjust class weight for imbalance
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'scale_pos_weight': [scale_pos_weight]
}

# Initialize XGBoost classifier
model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)

# Perform GridSearchCV
grid_search = GridSearchCV(model, param_grid, scoring='recall', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
}

print("Best Hyperparameters:", grid_search.best_params_)
print("Performance Metrics:", metrics)
