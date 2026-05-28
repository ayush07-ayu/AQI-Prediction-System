import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

def generate_training_data():
    """Generate comprehensive training data"""
    np.random.seed(42)
    
    # Generate synthetic but realistic data
    n_samples = 50000
    
    # Generate features with realistic correlations
    pm25 = np.random.uniform(10, 400, n_samples)
    pm10 = pm25 * np.random.uniform(1.0, 1.5, n_samples)
    no2 = pm25 * np.random.uniform(0.2, 0.4, n_samples)
    so2 = pm25 * np.random.uniform(0.05, 0.15, n_samples)
    co = pm25 * np.random.uniform(0.02, 0.06, n_samples)
    o3 = pm25 * np.random.uniform(0.1, 0.3, n_samples)
    
    temperature = np.random.uniform(10, 40, n_samples)
    humidity = np.random.uniform(30, 90, n_samples)
    wind_speed = np.random.uniform(0, 30, n_samples)
    vehicle_count = np.random.uniform(5000, 80000, n_samples)
    industrial_index = np.random.randint(1, 11, n_samples)
    
    # Calculate AQI based on weighted formula
    aqi = (pm25 * 0.6 + pm10 * 0.3 + no2 * 0.05 + 
           so2 * 0.02 + co * 0.01 + o3 * 0.02 + 
           (40 - temperature) * 0.5 + humidity * 0.1 + 
           vehicle_count / 1000 + industrial_index * 3)
    
    # Add some noise
    aqi += np.random.normal(0, 10, n_samples)
    aqi = np.clip(aqi, 0, 500)
    
    # Create DataFrame
    df = pd.DataFrame({
        'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2,
        'CO': co, 'O3': o3, 'Temperature': temperature,
        'Humidity': humidity, 'Wind_Speed': wind_speed,
        'Vehicle_Count': vehicle_count, 'Industrial_Index': industrial_index,
        'AQI': aqi
    })
    
    return df

def train_and_save_model():
    """Train Random Forest model and save to disk"""
    
    print("Generating training data...")
    df = generate_training_data()
    
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 
                'Temperature', 'Humidity', 'Wind_Speed', 
                'Vehicle_Count', 'Industrial_Index']
    
    X = df[features]
    y = df['AQI']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {features}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Random Forest
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Save model
    model_path = "aqi_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    return model, r2, mae

if __name__ == "__main__":
    train_and_save_model()