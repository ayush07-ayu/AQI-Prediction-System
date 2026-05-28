import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="India Pollution Analytics Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* KPI Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(255,255,255,0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Custom colors for different pollution levels */
    .good { color: #00ff00; }
    .satisfactory { color: #8bc34a; }
    .moderate { color: #ffeb3b; }
    .poor { color: #ff9800; }
    .very-poor { color: #ff5722; }
    .severe { color: #f44336; }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv("pollution_data.csv")

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    return df
    
    # Generate dates from 2018 to 2026
    dates = pd.date_range(start='2018-01-01', end='2026-01-31', freq='D')
    
    cities_data = {
        'Delhi': {'state': 'Delhi', 'lat': 28.6139, 'lon': 77.2090, 'base_aqi': 350},
        'Mumbai': {'state': 'Maharashtra', 'lat': 19.0760, 'lon': 72.8777, 'base_aqi': 180},
        'Bengaluru': {'state': 'Karnataka', 'lat': 12.9716, 'lon': 77.5946, 'base_aqi': 120},
        'Chennai': {'state': 'Tamil Nadu', 'lat': 13.0827, 'lon': 80.2707, 'base_aqi': 140},
        'Kolkata': {'state': 'West Bengal', 'lat': 22.5726, 'lon': 88.3639, 'base_aqi': 200},
        'Gurugram': {'state': 'Haryana', 'lat': 28.4595, 'lon': 77.0266, 'base_aqi': 370},
        'Faridabad': {'state': 'Haryana', 'lat': 28.4089, 'lon': 77.3178, 'base_aqi': 360},
        'Noida': {'state': 'Uttar Pradesh', 'lat': 28.5355, 'lon': 77.3910, 'base_aqi': 355},
        'Jaipur': {'state': 'Rajasthan', 'lat': 26.9124, 'lon': 75.7873, 'base_aqi': 190},
        'Lucknow': {'state': 'Uttar Pradesh', 'lat': 26.8467, 'lon': 80.9462, 'base_aqi': 220},
        'Pune': {'state': 'Maharashtra', 'lat': 18.5204, 'lon': 73.8567, 'base_aqi': 150},
        'Chandigarh': {'state': 'Chandigarh', 'lat': 30.7333, 'lon': 76.7794, 'base_aqi': 210},
        'Amritsar': {'state': 'Punjab', 'lat': 31.6340, 'lon': 74.8726, 'base_aqi': 240}
    }
    
    data_list = []
    
    for date in dates:
        for city, info in cities_data.items():
            # Seasonal variation (winter: Nov-Feb higher, monsoon: Jun-Sep lower)
            month = date.month
            season_factor = 1.0
            if month in [11, 12, 1, 2]:  # Winter
                season_factor = 1.4 if city in ['Delhi', 'Gurugram', 'Faridabad', 'Noida', 'Amritsar', 'Chandigarh'] else 1.2
            elif month in [6, 7, 8, 9]:  # Monsoon
                season_factor = 0.7
            elif month in [3, 4, 5]:  # Summer
                season_factor = 0.9
            
            # Festival effect (Oct-Nov)
            festival = 'Yes' if month in [10, 11] else 'No'
            festival_factor = 1.15 if festival == 'Yes' else 1.0
            
            # Crop burning effect (Oct-Nov in North India)
            crop_burning = 'High' if (month in [10, 11] and city in ['Delhi', 'Gurugram', 'Faridabad', 'Noida', 'Amritsar', 'Chandigarh']) else \
                          'Medium' if (month in [10, 11] and city in ['Lucknow', 'Jaipur']) else 'Low'
            crop_factor = 1.3 if crop_burning == 'High' else 1.1 if crop_burning == 'Medium' else 1.0
            
            # Yearly trend (increasing pollution)
            year_factor = 1 + (date.year - 2018) * 0.03
            
            # Calculate AQI
            base_aqi = info['base_aqi']
            aqi = base_aqi * season_factor * festival_factor * crop_factor * year_factor
            aqi = np.random.normal(aqi, aqi * 0.1)  # Add some randomness
            aqi = max(0, min(500, aqi))  # Clip to 0-500 range
            
            # Determine AQI category
            if aqi <= 50:
                aqi_category = 'Good'
            elif aqi <= 100:
                aqi_category = 'Satisfactory'
            elif aqi <= 200:
                aqi_category = 'Moderate'
            elif aqi <= 300:
                aqi_category = 'Poor'
            elif aqi <= 400:
                aqi_category = 'Very Poor'
            else:
                aqi_category = 'Severe'
            
            # Calculate other pollutants based on AQI
            pm25 = aqi * np.random.uniform(0.6, 0.8)
            pm10 = aqi * np.random.uniform(0.8, 1.2)
            no2 = aqi * np.random.uniform(0.15, 0.25)
            so2 = aqi * np.random.uniform(0.05, 0.1)
            co = aqi * np.random.uniform(0.02, 0.04)
            o3 = aqi * np.random.uniform(0.08, 0.15)
            
            # Weather conditions
            temp = 25 + 15 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3)
            temp = max(5, min(45, temp))
            
            humidity = 60 + 25 * np.sin(2 * np.pi * (month - 7) / 12) + np.random.normal(0, 10)
            humidity = max(20, min(100, humidity))
            
            wind_speed = 10 + 5 * np.random.random()
            rainfall = max(0, 150 * np.random.exponential(0.5) if month in [6,7,8] else 20 * np.random.exponential(0.5))
            
            # Human activities
            vehicle_count = 25000 + np.random.randint(-5000, 10000)
            industrial_index = np.random.randint(5, 10)
            population_density = np.random.randint(5000, 25000)
            construction_activity = np.random.randint(3, 10)
            
            # Traffic level
            if vehicle_count > 35000:
                traffic_level = 'High'
            elif vehicle_count > 25000:
                traffic_level = 'Medium'
            else:
                traffic_level = 'Low'
            
            # Future AQI (next day)
            future_aqi = aqi * np.random.uniform(0.95, 1.05)
            future_aqi = max(0, min(500, future_aqi))
            
            # AQI trend
            if future_aqi > aqi * 1.02:
                aqi_trend = 'Increasing'
            elif future_aqi < aqi * 0.98:
                aqi_trend = 'Decreasing'
            else:
                aqi_trend = 'Stable'
            
            data_list.append({
                'Date': date,
                'Year': date.year,
                'Month': month,
                'Day': date.day,
                'City': city,
                'State': info['state'],
                'Latitude': info['lat'],
                'Longitude': info['lon'],
                'AQI': round(aqi, 1),
                'AQI_Category': aqi_category,
                'PM2.5': round(pm25, 1),
                'PM10': round(pm10, 1),
                'NO2': round(no2, 1),
                'SO2': round(so2, 1),
                'CO': round(co, 2),
                'O3': round(o3, 1),
                'Temperature': round(temp, 1),
                'Humidity': round(humidity, 1),
                'Wind_Speed': round(wind_speed, 1),
                'Rainfall': round(rainfall, 1),
                'Pressure': round(1010 + np.random.normal(0, 5), 1),
                'Vehicle_Count': vehicle_count,
                'Industrial_Index': industrial_index,
                'Population_Density': population_density,
                'Construction_Activity': construction_activity,
                'Festival_Season': festival,
                'Crop_Burning_Impact': crop_burning,
                'Traffic_Level': traffic_level,
                'Future_AQI': round(future_aqi, 1),
                'AQI_Trend': aqi_trend
            })
    
    df = pd.DataFrame(data_list)
    return df

# Main app
def main():
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>🌍 India Pollution Analytics Dashboard</h1>
        <p>Real-time Air Quality Monitoring | Predictive Analytics | Environmental Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading pollution data..."):
        df = load_data()
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🔍 Filters")
        
        # State filter
        states = ['All'] + sorted(df['State'].unique().tolist())
        selected_state = st.selectbox("Select State", states)
        
        # City filter based on state selection
        if selected_state != 'All':
            cities = ['All'] + sorted(df[df['State'] == selected_state]['City'].unique().tolist())
        else:
            cities = ['All'] + sorted(df['City'].unique().tolist())
        
        selected_city = st.selectbox("Select City", cities)
        
        # AQI category filter
        aqi_categories = ['All'] + sorted(df['AQI_Category'].unique().tolist())
        selected_category = st.selectbox("AQI Category", aqi_categories)
        
        # Date range filter
        st.markdown("### 📅 Date Range")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Apply filters
        st.markdown("---")
        st.markdown("### 📊 Dashboard Info")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Cities Covered", df['City'].nunique())
        st.metric("Date Range", f"{min_date.year} - {max_date.year}")
    
    # Filter data
    filtered_df = df.copy()
    
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['City'] == selected_city]
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['AQI_Category'] == selected_category]
    
    filtered_df = filtered_df[(filtered_df['Date'].dt.date >= start_date) & 
                               (filtered_df['Date'].dt.date <= end_date)]
    
    # KPI Cards
    st.markdown("## 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_aqi = filtered_df['AQI'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌫️ Average AQI</h3>
            <h2>{avg_aqi:.1f}</h2>
            <p>Air Quality Index</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        max_pm25 = filtered_df['PM2.5'].max()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚫ Highest PM2.5</h3>
            <h2>{max_pm25:.1f}</h2>
            <p>μg/m³</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_temp = filtered_df['Temperature'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌡️ Avg Temperature</h3>
            <h2>{avg_temp:.1f}°C</h2>
            <p>Average Temp</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_humidity = filtered_df['Humidity'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>💧 Humidity Level</h3>
            <h2>{avg_humidity:.1f}%</h2>
            <p>Relative Humidity</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 1: AQI Trend and PM2.5
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 AQI Trend Over Time")
        fig_aqi = px.line(
            filtered_df.groupby('Date')['AQI'].mean().reset_index(),
            x='Date', y='AQI',
            title="Air Quality Index Trend",
            color_discrete_sequence=['#667eea']
        )
        fig_aqi.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date",
            yaxis_title="AQI"
        )
        st.plotly_chart(fig_aqi, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 PM2.5 Trend")
        fig_pm25 = px.line(
            filtered_df.groupby('Date')['PM2.5'].mean().reset_index(),
            x='Date', y='PM2.5',
            title="PM2.5 Concentration Trend",
            color_discrete_sequence=['#f093fb']
        )
        fig_pm25.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date",
            yaxis_title="PM2.5 (μg/m³)"
        )
        st.plotly_chart(fig_pm25, use_container_width=True)
    
    # Row 2: Pollution Comparison
    st.markdown("### 🧪 Pollution Parameters Comparison")
    
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    pollutant_avg = filtered_df[pollutants].mean().reset_index()
    pollutant_avg.columns = ['Pollutant', 'Value']
    
    fig_compare = px.bar(
        pollutant_avg,
        x='Pollutant', y='Value',
        title="Average Pollution Levels by Parameter",
        color='Pollutant',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_compare.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Pollutant",
        yaxis_title="Concentration"
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Row 3: AQI Category Distribution and Weather Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 AQI Category Distribution")
        category_counts = filtered_df['AQI_Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        fig_categories = px.pie(
            category_counts,
            values='Count', names='Category',
            title="Air Quality Categories",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_categories.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_categories, use_container_width=True)
    
    with col2:
        st.markdown("### 🌤️ Weather Analysis")
        
        weather_data = filtered_df.groupby('Date')[['Temperature', 'Humidity', 'Rainfall', 'Wind_Speed']].mean().reset_index()
        weather_data = weather_data.tail(30)  # Last 30 days
        
        fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_weather.add_trace(
            go.Scatter(x=weather_data['Date'], y=weather_data['Temperature'], name="Temperature (°C)", line=dict(color='#ff6b6b')),
            secondary_y=False
        )
        
        fig_weather.add_trace(
            go.Scatter(x=weather_data['Date'], y=weather_data['Humidity'], name="Humidity (%)", line=dict(color='#4ecdc4')),
            secondary_y=True
        )
        
        fig_weather.update_layout(
            title="Temperature vs Humidity Trends",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date"
        )
        
        st.plotly_chart(fig_weather, use_container_width=True)
    
    # Row 4: Traffic vs AQI and Industrial Index vs AQI
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚗 Traffic Level vs AQI")
        
        traffic_aqi = filtered_df.groupby('Traffic_Level')['AQI'].mean().reset_index()
        
        fig_traffic = px.bar(
            traffic_aqi,
            x='Traffic_Level', y='AQI',
            title="Average AQI by Traffic Level",
            color='Traffic_Level',
            color_discrete_sequence=['#ffa07a', '#f4a460', '#cd5c5c']
        )
        fig_traffic.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Traffic Level",
            yaxis_title="Average AQI"
        )
        st.plotly_chart(fig_traffic, use_container_width=True)
    
    with col2:
        st.markdown("### 🏭 Industrial Index vs AQI")
        
        industry_aqi = filtered_df.groupby('Industrial_Index')['AQI'].mean().reset_index()
        
        fig_industry = px.scatter(
            filtered_df,
            x='Industrial_Index', y='AQI',
            title="Industrial Activity Impact on AQI",
            trendline="ols",
            color='City',
            opacity=0.6
        )
        fig_industry.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Industrial Index (1-10)",
            yaxis_title="AQI"
        )
        st.plotly_chart(fig_industry, use_container_width=True)
    
    # Row 5: Future AQI Prediction
    st.markdown("### 🔮 Future AQI Prediction")
    
    future_data = filtered_df.groupby('Date')[['AQI', 'Future_AQI']].mean().reset_index().tail(60)
    
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=future_data['Date'], y=future_data['AQI'], 
                                    name='Current AQI', line=dict(color='#667eea', width=2)))
    fig_future.add_trace(go.Scatter(x=future_data['Date'], y=future_data['Future_AQI'], 
                                    name='Predicted Future AQI', line=dict(color='#f093fb', width=2, dash='dash')))
    
    fig_future.update_layout(
        title="Current vs Predicted AQI (Next Day Forecast)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode='x unified'
    )
    st.plotly_chart(fig_future, use_container_width=True)
    
    # Row 6: AQI Trend Analysis
    st.markdown("### 📊 AQI Trend Analysis")
    
    trend_counts = filtered_df['AQI_Trend'].value_counts().reset_index()
    trend_counts.columns = ['Trend', 'Count']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig_trend_pie = px.pie(
            trend_counts,
            values='Count', names='Trend',
            title="AQI Trend Distribution",
            color_discrete_sequence=['#4ecdc4', '#ff6b6b', '#95e77e']
        )
        fig_trend_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_trend_pie, use_container_width=True)
    
    with col2:
        trend_by_city = filtered_df.groupby('City')['AQI_Trend'].value_counts().unstack().fillna(0)
        fig_trend_bar = px.bar(
            trend_by_city.reset_index(),
            x='City', y=['Increasing', 'Stable', 'Decreasing'],
            title="AQI Trends by City",
            barmode='group'
        )
        fig_trend_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="City",
            yaxis_title="Number of Days"
        )
        st.plotly_chart(fig_trend_bar, use_container_width=True)
    
    # Map Visualization
    st.markdown("### 🗺️ Interactive Pollution Map")
    
    map_data = filtered_df.groupby(['City', 'Latitude', 'Longitude'])[['AQI', 'PM2.5', 'PM10']].mean().reset_index()
    
    fig_map = px.scatter_mapbox(
        map_data,
        lat="Latitude",
        lon="Longitude",
        size="AQI",
        color="AQI",
        hover_name="City",
        hover_data={'PM2.5': True, 'PM10': True, 'AQI': True},
        color_continuous_scale="RdYlGn_r",
        size_max=40,
        zoom=3,
        title="Air Quality Index Across India"
    )
    
    fig_map.update_layout(
        mapbox_style="open-street-map",
        mapbox_accesstoken=None,
        height=500,
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Raw Data Preview
    st.markdown("### 📋 Raw Dataset Preview")
    
    show_data = st.checkbox("Show Full Dataset")
    
    if show_data:
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button(
            label="📥 Download Dataset as CSV",
            data=filtered_df.to_csv(index=False),
            file_name=f"india_pollution_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.dataframe(filtered_df.head(100), use_container_width=True)
        st.info("Showing first 100 rows. Check 'Show Full Dataset' to view all records.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🌍 India Pollution Analytics Dashboard | Data Source: Central Pollution Control Board (CPCB) | 
        Last Updated: {} | Dashboard by: Environmental Data Analytics Team</p>
    </div>
    """.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()