import xgboost as xgb
import matplotlib.pyplot as plt

# Train the model first
model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=5)
model.fit(X_train_resampled, y_train_resampled)  # Make sure you're using the trained data

# Get feature importance
feature_importance = model.feature_importances_

# Plot the feature importance
plt.figure(figsize=(10,5))
plt.barh(X_train_resampled.columns, feature_importance)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance in XGBoost")
plt.show()
