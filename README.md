AI-Based Training Program Optimization

This repository contains a Streamlit app that clusters athletes based on their workout data and suggests personalized training programs. The goal is to provide coaches and trainers with data-driven insights to improve athlete performance while minimizing the risk of fatigue and injury.

Features

Athlete Data Clustering: Uses K-Means clustering to group athletes based on workout load, fatigue index, recovery time, and performance improvement.

Interactive Input: Allows users to input new athlete data and predicts the cluster they belong to.

Personalized Recommendations: Provides tailored workout recommendations based on the predicted cluster.

Sample Dataset: Contains a synthetic dataset of 5000 athletes for demonstration purposes.

How It Works

Data Input:

The app reads athlete data from an Excel file (athlete_data_5000.xlsx) hosted in this repository.

Users can input new athlete data through sliders and number inputs.

Clustering Model:

The app scales the input data and applies a pre-trained K-Means model with 3 clusters.

Based on the predicted cluster, the app suggests an appropriate training program.

Setup and Deployment

Running Locally

Clone the repository:

git clone https://github.com/MiamiCrypto/Training_Program_Optimization.git

Navigate to the project directory:

cd Training_Program_Optimization

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Deploying on Streamlit Cloud

Fork this repository or create your own repository containing the app.py file and the dataset.

Go to Streamlit Cloud and create a new app deployment linked to your GitHub repository.

Follow the instructions to deploy the app.

Dataset

The synthetic dataset athlete_data_5000.xlsx contains the following fields:

athlete_id: Unique identifier for each athlete.

age: Age of the athlete.

workout_load: Total weekly workload of the athlete.

fatigue_index: Subjective fatigue level on a scale of 1 to 10.

recovery_time: Number of days required for recovery.

performance_improvement: Improvement ratio in performance.

Technologies Used

Streamlit: For building the web application.

Pandas: For data manipulation.

NumPy: For numerical computations.

Scikit-Learn: For K-Means clustering and data scaling.

Matplotlib/Seaborn: (Optional) For additional data visualizations.

Future Enhancements

Incorporate additional clustering techniques and compare results.

Add visualizations to show the distribution of clusters.

Enable dynamic updating of the dataset with new athlete entries.

Allow export of personalized workout plans.

Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

License

This project is licensed under the MIT License.

Contact

For any questions or inquiries, please contact Ramon Castro at miamicryptolabs@gmail.com
