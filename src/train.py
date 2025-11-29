# src/train.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.data_loader import load_dataset

# Data
X, y = load_dataset()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess
preprocess = ColumnTransformer([
    ("scaler", StandardScaler(), X.columns)
])

# Model
model = Pipeline([
    ("prep", preprocess),
    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42))
])

model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n===== Training Complete =====")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")

# Save
joblib.dump(model, "house_price_model.joblib")
print("\nModel saved → house_price_model.joblib")
