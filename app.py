import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
from io import BytesIO
import plotly.express as px
from fpdf import FPDF

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

    # Predict clusters and assign to DataFrame
    df['cluster'] = kmeans.predict(scaled_features)

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

    # Cluster insights section
    st.subheader("Cluster Insights")
    cluster_summary = df.groupby('cluster').mean()[features]
    st.write(cluster_summary)

    # Data visualization
    st.subheader("Cluster Visualization")
    fig = px.scatter(df, x='workout_load', y='performance_improvement', color=df['cluster'].astype(str),
                     title="Workout Load vs Performance Improvement by Cluster",
                     labels={"cluster": "Cluster"})
    st.plotly_chart(fig)

    # Downloadable PDF report
    st.subheader("Download Report")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="AI-Based Training Program Optimization Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
        pdf.cell(200, 10, txt=f"Workout Load: {workout_load}", ln=True)
        pdf.cell(200, 10, txt=f"Fatigue Index: {fatigue_index}", ln=True)
        pdf.cell(200, 10, txt=f"Recovery Time: {recovery_time}", ln=True)
        pdf.cell(200, 10, txt=f"Performance Improvement: {performance_improvement}", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Cluster: {cluster}", ln=True)
        if cluster == 0:
            pdf.cell(200, 10, txt="Suggested Training Program: Focus on endurance and stamina-building exercises.", ln=True)
        elif cluster == 1:
            pdf.cell(200, 10, txt="Suggested Training Program: Emphasize strength training and muscle recovery.", ln=True)
        else:
            pdf.cell(200, 10, txt="Suggested Training Program: Balance intensity workouts with adequate recovery time.", ln=True)
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        st.download_button(label="Download PDF", data=pdf_output, file_name="training_program_report.pdf", mime="application/pdf")

    # File upload for new dataset
    st.subheader("Upload New Athlete Dataset")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        new_df = pd.read_excel(uploaded_file)
        st.write("Uploaded Dataset:")
        st.dataframe(new_df.head())
        new_scaled_features = scaler.transform(new_df[features])
        new_df['cluster'] = kmeans.predict(new_scaled_features)
        st.write("Clustered Dataset:")
        st.dataframe(new_df)

if __name__ == "__main__":
    main()
