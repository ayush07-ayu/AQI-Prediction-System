import streamlit as st
import pandas as pd

df = pd.read_csv("pollution_data.csv")

st.title("🌍 India Pollution Dashboard")

# city filter (agar column hai)
if "City" in df.columns:
    city = st.selectbox("City select karo", df["City"].unique())
    df = df[df["City"] == city]

st.dataframe(df)
st.line_chart(df.select_dtypes(include="number"))