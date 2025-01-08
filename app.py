import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
from io import BytesIO

# Load the dataset from GitHub raw URL
url = "https://github.com/MiamiCrypto/Training_Program_Optimization/raw/refs/heads/master/athlete_data_5000.xlsx"
response = requests.get(url)
df = pd.read_excel(BytesIO(response.content))

# Set up Streamlit app
def main():
    st.title("AI-Based Training Program Optimization")
    st.write("This app clusters athletes based on their workout data and suggests personalized training programs.")

    # Display sample data
    st.subheader("Sample Athlete Data")
    st.dataframe(df.head())

    # User input section
    st.subheader("Enter New Athlete Data")
    age = st.number_input("Age", min_value=15, max_value=50, value=25)
    workout_load = st.slider("Workout Load", min_value=30, max_value=200, value=100)
    fatigue_index = st.slider("Fatigue Index", min_value=1.0, max_value=10.0, value=5.0)
    recovery_time = st.slider("Recovery Time (Days)", min_value=1, max_value=7, value=3)
    performance_improvement = st.slider("Performance Improvement", min_value=0.5, max_value=5.0, value=1.5)

    # Prepare input data
    input_data = pd.DataFrame({
        'age': [age],
        'workout_load': [workout_load],
        'fatigue_index': [fatigue_index],
        'recovery_time': [recovery_time],
        'performance_improvement': [performance_improvement]
    })

    # Feature selection and scaling
    features = ['workout_load', 'fatigue_index', 'recovery_time', 'performance_improvement']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled_features)

    # Predict cluster for new data
    scaled_input = scaler.transform(input_data[features])
    cluster = kmeans.predict(scaled_input)[0]

    # Display the predicted cluster
    st.subheader("Predicted Cluster")
    st.write(f"The new athlete belongs to Cluster {cluster}.")

    # Suggested training program based on cluster
    st.subheader("Suggested Training Program")
    if cluster == 0:
        st.write("Focus on endurance and stamina-building exercises.")
    elif cluster == 1:
        st.write("Emphasize strength training and muscle recovery.")
    else:
        st.write("Balance intensity workouts with adequate recovery time.")

if __name__ == "__main__":
    main()
