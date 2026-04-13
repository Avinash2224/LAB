import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

# Load wine dataset
from sklearn.datasets import fetch_openml
data = fetch_openml(name='wine-quality-red', version=1, as_frame=True)
X = data.data
y = data.target.astype(float)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")
print("Run completed by: Avinash Alash - 2022BCS0021")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metrics
metrics = {"mse": mse, "r2": r2}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Model and metrics saved!")