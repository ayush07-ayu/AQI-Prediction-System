#  India Pollution Analytics Dashboard

A powerful Machine Learning and Data Analytics dashboard built using **Python, Streamlit, Plotly, and Scikit-learn** to analyze and predict Air Quality Index (AQI) across major Indian cities.

The project provides real-time style pollution analytics, interactive visualizations, predictive machine learning models, and environmental insights through a modern and responsive dashboard.

---

#  Features

##  Interactive Analytics Dashboard

* AQI trend analysis over time
* PM2.5 and PM10 pollution monitoring
* Pollution category distribution
* Weather vs AQI correlation analysis
* Traffic and industrial impact visualization
* Interactive India pollution map

---

##  Machine Learning AQI Prediction

* AQI prediction using Random Forest Regression
* Trained ML model saved using Joblib
* Interactive prediction module
* Feature importance analysis
* Actual vs Predicted AQI comparison
* Real-time prediction interface

---

##  Advanced Data Visualization

* Plotly interactive charts
* Dynamic filtering system
* State and city-based analytics
* Time-series pollution trends
* Environmental insights dashboard

---

# 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* Scikit-learn
* Joblib

---

#  Machine Learning Model

The project uses:

## Random Forest Regressor

Used for AQI prediction based on:

* PM2.5
* PM10
* NO2
* SO2
* CO
* O3
* Temperature
* Humidity
* Wind Speed
* Vehicle Count
* Industrial Index

### Model Performance

* High R² Score
* Low Mean Absolute Error (MAE)
* Optimized for accurate AQI forecasting

---

# 📂 Project Structure

```bash
india-pollution-dashboard/
│
├── app.py
├── train_model.py
├── aqi_model.pkl
├── pollution_data.csv
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/india-pollution-dashboard.git
cd india-pollution-dashboard
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  Run The Application

```bash
streamlit run app.py
```

---

#  Train The ML Model

```bash
python train_model.py
```

This will generate:

```bash
aqi_model.pkl
```

---

#  Dashboard Modules

##  Pollution Analytics

Provides deep insights into:

* AQI patterns
* Pollution levels
* Seasonal effects
* Weather impact

---

##  Interactive India Pollution Map

Visual representation of AQI across Indian cities using Mapbox and Plotly.

---

##  AQI Prediction System

Users can enter pollution parameters and get predicted AQI instantly using the trained machine learning model.

---

#  Future Improvements

* Real-time AQI API integration
* Deep Learning models (LSTM)
* Live pollution monitoring
* Forecasting for next 7 days
* Mobile responsive optimization
* User authentication system

---

#  Author

Ayush Sati

---

# License

This project is created for educational, research, and portfolio purposes.

---

#  Support

If you like this project:

* Star this repository
* Fork the project
* Share feedback
* Connect on LinkedIn

---

# Environmental Vision

This project aims to spread awareness about air pollution and demonstrate how Machine Learning and Data Analytics can help in understanding environmental challenges and building smarter solutions for a cleaner future.
