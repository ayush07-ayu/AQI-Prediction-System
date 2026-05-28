import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv("pollution_data.csv")

# Features
features = [
    'PM2.5',
    'PM10',
    'NO2',
    'SO2',
    'CO',
    'O3',
    'Temperature',
    'Humidity',
    'Wind_Speed',
    'Vehicle_Count',
    'Industrial_Index'
]

# Target
target = 'AQI'

# Input and output
X = df[features]
y = df[target]

# Handle missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

print("Model Trained Successfully")
print("R2 Score:", round(score, 3))

# Save model
joblib.dump(model, "aqi_model.pkl")

print("Model saved as aqi_model.pkl")