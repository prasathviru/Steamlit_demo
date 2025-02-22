import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Simple Dataset (House Prices)
data = {
    "Area (sq ft)": [500, 600, 700, 800, 900, 1000],
    "Price ($)": [15000, 18000, 21000, 24000, 27000, 30000]
}
df = pd.DataFrame(data)

# Streamlit UI
st.title("🏡 Simple House Price Prediction")
st.write("This app predicts house prices based on area (sq ft).")

# Show Dataset
st.write("### Dataset (Used for Training):")
st.write(df)

# Train Model
X = df[["Area (sq ft)"]]
y = df["Price ($)"]
model = LinearRegression().fit(X, y)

# User Input
area = st.number_input("Enter House Area (sq ft):", min_value=100, max_value=2000, step=50)

# Predict Price
if st.button("Predict Price"):
    price = model.predict([[area]])[0]
    st.write(f"### Predicted Price: **${price:.2f}**")
