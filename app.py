import streamlit as st
import pandas as pd
import pickle
import numpy as np
from streamlit_option_menu import option_menu
import re
import os

# Set page configuration
st.set_page_config(page_title="Sleep Disorder Prediction App", layout="wide")

# Custom CSS for consistent sleep-themed styling
st.markdown("""
    <style>
    /* General App Styling */
    body {
        background-color: #F3F4F6; /* Light gray background */
    }
    .stNumberInput input, .stSelectbox select {
        border-radius: 5px;
        padding: 8px;
        border: 1px solid #D1D5DB; /* Gray border */
        font-size: 16px;
        background-color: #FFFFFF;
    }
    .stButton button {
        background-color: #1E3A8A; /* Deep blue */
        color: #FFFFFF;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #6B7280; /* Soft purple */
    }
    .stForm {
        background-color: #F3F4F6; /* Light gray */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .subheader {
        font-size: 18px;
        color: #1F2937; /* Dark gray */
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: bold; /* Make subheaders bold */
    }
    .help-text {
        font-size: 14px;
        color: #6B7280; /* Soft purple */
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 24px;
        color: #1F2937; /* Dark gray */
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        font-weight: bold; /* Make section titles bold */
    }
    .section-title img {
        margin-right: 10px;
    }
    /* Page Banners */
    .banner-home {
        background: linear-gradient(135deg, #1E3A8A, #3B82F6); /* Deep blue to light blue */
        color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .banner-eda {
        background: linear-gradient(135deg, #1E3A8A, #6B7280); /* Deep blue to soft purple */
        color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .banner-training {
        background: linear-gradient(135deg, #1E3A8A, #2563EB); /* Deep blue to vibrant blue */
        color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .banner-results {
        background: linear-gradient(135deg, #1E3A8A, #4B5563); /* Deep blue to grayish purple */
        color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Table Styling */
    .stDataFrame {
        border: 1px solid #D1D5DB; /* Gray border */
        border-radius: 5px;
        overflow: hidden;
    }
    .stDataFrame table {
        width: 100%;
        border-collapse: collapse;
    }
    .stDataFrame th {
        background-color: #DBEAFE; /* Light blue */
        color: #1F2937; /* Dark gray */
        font-weight: bold;
    }
    .stDataFrame tr:nth-child(even) {
        background-color: #F9FAFB; /* Off-white */
    }
    .stDataFrame tr:hover {
        background-color: #DBEAFE; /* Light blue */
    }
    /* Sidebar Styling */
    .css-1lcbmhc { /* Streamlit sidebar class */
        background-color: #F3F4F6; /* Light gray */
    }
    .css-1lcbmhc .stSelectbox, .css-1lcbmhc .stButton {
        color: #1F2937; /* Dark gray */
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessors globally
@st.cache_resource
def load_model_files():
    with open("models/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/training_columns.pkl", "rb") as f:
        training_columns = pickle.load(f)
    with open("models/accuracy.txt", "r") as f:
        accuracy = float(f.read())
    with open("models/classification_report.txt", "r") as f:
        class_report = f.read()
    return rf_model, label_encoder, scaler, training_columns, accuracy, class_report

rf_model, label_encoder, scaler, training_columns, accuracy, class_report = load_model_files()

# Sidebar navigation
with st.sidebar:
    st.markdown('<h2 style="color: #1E3A8A;"> Sleep Disorder Prediction</h2>', unsafe_allow_html=True)
    selected = option_menu(
        "Main Menu",
        ["Home", "EDA", "Model Training", "Results"],
        icons=["house", "bar-chart", "gear", "clipboard-data"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#F3F4F6"},
            "icon": {"color": "#1E3A8A"},
            "nav-link": {"color": "#1F2937", "--hover-color": "#DBEAFE"},
            "nav-link-selected": {"background-color": "#1E3A8A", "color": "#FFFFFF"}
        }
    )

# Home Page
if selected == "Home":
    st.markdown('<div class="banner-home"><h1> Sleep Disorder Prediction App</h1><p>Explore the Sleep Health and Lifestyle Dataset to uncover patterns and predict sleep disorders.</p></div>', unsafe_allow_html=True)

    # Dataset Overview
    with st.container():
        st.markdown('<div class="section-title">📚 Dataset Overview</div>', unsafe_allow_html=True)
        with st.expander("View Details", expanded=True):
            st.markdown("""
                The **Sleep Health and Lifestyle Dataset** contains 374 rows and 13 columns, providing insights into various factors affecting sleep and daily lifestyle habits. It includes information on demographics, sleep metrics, lifestyle factors, cardiovascular health, and sleep disorders. The dataset covers key variables such as age, gender, occupation, sleep duration, quality of sleep, physical activity, stress levels, BMI, blood pressure, heart rate, daily steps, and sleep disorders like insomnia and sleep apnea.
            """)
            show_features = st.checkbox("Show Key Features", value=False)
            if show_features:
                st.markdown("""
                    **Key Features:**
                    - **Sleep Metrics**: Sleep duration and quality.
                    - **Lifestyle Factors**: Physical activity levels, stress levels, and BMI categories.
                    - **Cardiovascular Health**: Blood pressure and heart rate measurements.
                    - **Sleep Disorders**: Presence of conditions like insomnia and sleep apnea.
                """)

    # Load dataset
    data = pd.read_excel("sleep_data.xlsx", engine="openpyxl")

    # Data Preview
    with st.container():
        st.markdown('<div class="section-title">📊 Dataset Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader">First 8 Records of Dataset</div>', unsafe_allow_html=True)
        st.dataframe(data.head(8), use_container_width=True)
        st.download_button(
            label="Download Full Dataset",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name="sleep_data.csv",
            mime="text/csv",
            key="download_dataset"
        )

    # Summary Statistics
    with st.container():
        st.markdown('<div class="section-title">🔢 Summary Statistics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="subheader">Categorical Variables</div>', unsafe_allow_html=True)
            cat_summary_df = pd.read_csv("eda_results/categorical_summary.csv", index_col=0)
            st.dataframe(cat_summary_df, use_container_width=True)

        with col2:
            st.markdown('<div class="subheader">Numeric Variables</div>', unsafe_allow_html=True)
            num_summary_df = pd.read_csv("eda_results/numeric_summary.csv", index_col=0)
            st.dataframe(num_summary_df, use_container_width=True)

    # EDA Results
    with st.container():
        st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        eda_results_path = "eda_results/"
        eda_files = [f for f in os.listdir(eda_results_path) if f.endswith(".csv") and f != "categorical_summary.csv" and f != "numeric_summary.csv"]
        
        for file in eda_files:
            with st.expander(f"{file.replace('.csv', '').replace('_', ' ').title()}"):
                df = pd.read_csv(os.path.join(eda_results_path, file), index_col=0)
                st.dataframe(df, use_container_width=True)

# EDA Page
elif selected == "EDA":
    st.markdown('<div class="banner-eda"><h1> Exploratory Data Analysis</h1><p>Visualize patterns in the Sleep Health and Lifestyle Dataset to understand sleep disorder factors.</p></div>', unsafe_allow_html=True)

    plot_files = [
        "gender_vs_quality_sleep.png",
        "sleep_quality_distribution.png",
        "sleep_quality_by_occupation_heatmap.png",
        "correlation_heatmap.png",
        "sleep_quality_by_bmi.png",
        "sleep_quality_by_gender_bmi.png",
        "stress_level_by_occupation.png",
        "sleep_disorder_by_gender.png",
        "daily_steps_by_bmi.png",
        "activity_by_sleep_disorder.png",
        "sleep_vs_stress.png",
        "steps_vs_heart_rate.png",
        "health_risk_vs_quality.png",
        "sti_vs_quality_sleep.png"
    ]

    insights = [
        "Females tend to report higher quality of sleep (e.g., 9) more frequently than males, indicating a gender-based difference in sleep quality prevalence.",
        "The majority of people report sleep quality between 6 and 9, with the highest number (around 100) at quality 8, and a sharp decline at lower qualities (4-5).",
        "Engineers report the highest average sleep quality (8.4), while Sales Representatives have the lowest (4.0), highlighting occupation-related sleep quality variations.",
        "There is a strong positive correlation (0.88) between sleep duration and quality of sleep, suggesting that longer sleep duration significantly enhances sleep quality.",
        "Sleep quality decreases from 8 in the 'Normal' BMI category to 6 in the 'Obese' category, indicating a negative impact of increasing BMI on sleep quality.",
        "Females in the 'Normal' BMI category have the highest average sleep quality (8.3), while males in the 'Overweight' category have the lowest (6.1).",
        "Sales Representatives and Doctors exhibit the highest average stress levels (around 7-8), while Teachers and Engineers show the lowest (around 4-5).",
        "Males have a higher prevalence of 'No Disease' (around 140) compared to females (around 80), while females show a higher incidence of 'Sleep Apnea' (around 70).",
        "Individuals in the 'Overweight' category take significantly more daily steps (around 7000) compared to 'Obese' individuals (around 3000-4000), suggesting a potential correlation with activity levels.",
        "Individuals with 'No Disease' exhibit a wider range of physical activity levels (40-100) compared to those with 'Insomnia' (more concentrated around 40-60).",
        "Higher sleep quality (8-9) is more common with lower stress levels (3-4) and longer sleep durations (8-8.5 hours).",
        "There is no strong correlation between daily steps and heart rate, with heart rates stabilizing around 70-75 bpm across a wide range of step counts (3000-10000).",
        "Lower health risk scores (around 12-14) are associated with higher sleep quality (7-9) for both males and females, indicating a link between health risk and sleep.",
        "Higher sleep quality (8-9) is associated with a higher stress tolerance index (around 17-22.5), while lower sleep quality (4-5) corresponds to a lower index (around 10-12.5)."
    ]

    for plot_file, insight in zip(plot_files, insights):
        st.markdown(f'<div class="subheader">{plot_file.replace(".png", "").replace("_", " ").title()}</div>', unsafe_allow_html=True)
        st.image(f"plots/{plot_file}", use_container_width=True)
        st.write(insight)

# Model Training Page
elif selected == "Model Training":
    st.markdown('<div class="banner-training"><h1> Predict Sleep Disorders</h1><p>Enter patient details to predict the likelihood of sleep disorders using our trained model.</p></div>', unsafe_allow_html=True)

    # Create input form
    with st.form("prediction_form"):
        st.markdown('<div class="subheader">Patient Information</div>', unsafe_allow_html=True)
        st.markdown('<p class="help-text">Provide accurate details for the best prediction results. All fields are required.</p>', unsafe_allow_html=True)

        # Two-column layout
        col1, col2 = st.columns([1, 1])

        # Column 1: Demographics and Sleep Metrics
        with col1:
            st.markdown('<div class="subheader">Demographics & Sleep</div>', unsafe_allow_html=True)
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
                help="Enter age between 18 and 100 years."
            )
            sleep_duration = st.number_input(
                "Sleep Duration (hours)",
                min_value=1.0,
                max_value=12.0,
                value=7.0,
                step=0.1,
                help="Average hours of sleep per night (1.0 to 12.0)."
            )
            quality_of_sleep = st.number_input(
                "Quality of Sleep (1-10)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Rate sleep quality from 1 (poor) to 10 (excellent)."
            )
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                help="Select biological gender."
            )

        # Column 2: Health and Lifestyle Metrics
        with col2:
            st.markdown('<div class="subheader">Health & Lifestyle</div>', unsafe_allow_html=True)
            physical_activity = st.number_input(
                "Physical Activity Level (0-100)",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                help="Activity level score from 0 (inactive) to 100 (very active)."
            )
            stress_level = st.number_input(
                "Stress Level (1-10)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Rate stress level from 1 (low) to 10 (high)."
            )
            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=50,
                max_value=120,
                value=70,
                step=1,
                help="Resting heart rate in beats per minute (50-120)."
            )
            daily_steps = st.number_input(
                "Daily Steps",
                min_value=0,
                max_value=20000,
                value=5000,
                step=100,
                help="Average daily steps (0-20,000)."
            )

        # Full-width inputs for blood pressure and categorical fields
        st.markdown('<div class="subheader">Blood Pressure</div>', unsafe_allow_html=True)
        col_bp1, col_bp2 = st.columns([1, 1])
        with col_bp1:
            systolic = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=80.0,
                max_value=200.0,
                value=120.0,
                step=0.1,
                help="Systolic BP (top number, 80-200 mmHg)."
            )
        with col_bp2:
            diastolic = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                min_value=50.0,
                max_value=120.0,
                value=80.0,
                step=0.1,
                help="Diastolic BP (bottom number, 50-120 mmHg)."
            )

        st.markdown('<div class="subheader">Occupation & BMI</div>', unsafe_allow_html=True)
        occupation = st.selectbox(
            "Occupation",
            options=["Nurse", "Doctor", "Engineer", "Lawyer", "Teacher", "Accountant",
                     "Salesperson", "Software Engineer", "Scientist", "Sales Representative", "Manager"],
            help="Select current occupation."
        )
        bmi_category = st.selectbox(
            "BMI Category",
            options=["Normal", "Overweight", "Obese"],
            help="Select BMI category based on body mass index."
        )

        # Submit button
        submitted = st.form_submit_button("Predict")

    # Process inputs and make prediction
    if submitted:
        # Input validation
        if any([
            age < 18 or age > 100,
            sleep_duration < 1.0 or sleep_duration > 12.0,
            quality_of_sleep < 1 or quality_of_sleep > 10,
            physical_activity < 0 or physical_activity > 100,
            stress_level < 1 or stress_level > 10,
            heart_rate < 50 or heart_rate > 120,
            daily_steps < 0 or daily_steps > 20000,
            systolic < 80.0 or systolic > 200.0,
            diastolic < 50.0 or diastolic > 120.0
        ]):
            st.error("Please ensure all numerical inputs are within valid ranges.")
        else:
            # Create a dictionary with input data
            input_data = {
                "Age": age,
                "Sleep Duration": sleep_duration,
                "Quality of Sleep": quality_of_sleep,
                "Physical Activity Level": physical_activity,
                "Stress Level": stress_level,
                "Heart Rate": heart_rate,
                "Daily Steps": daily_steps,
                "Systolic": systolic,
                "Diastolic": diastolic,
                "Gender": gender,
                "Occupation": occupation,
                "BMI Category": bmi_category
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Preprocess numerical features
            num_features = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
                            "Stress Level", "Heart Rate", "Daily Steps", "Systolic", "Diastolic"]
            input_df[num_features] = scaler.transform(input_df[num_features])

            # Preprocess categorical features (one-hot encoding)
            cat_features = ["Gender", "Occupation", "BMI Category"]
            input_df = pd.get_dummies(input_df, columns=cat_features)

            # Ensure all columns from training are present
            for col in training_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[training_columns]  # Reorder columns to match training data

            # Make prediction
            with st.spinner("Making prediction..."):
                prediction = rf_model.predict(input_df)
                predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Display result
            st.markdown('<div class="subheader">Prediction Result</div>', unsafe_allow_html=True)
            if predicted_label == "No Disease":
                st.success(f"Predicted Sleep Disorder: {predicted_label}")
            else:
                st.error(f"Predicted Sleep Disorder: {predicted_label}")

# Results Page
elif selected == "Results":
    st.markdown('<div class="banner-results"><h1> Model Performance Results</h1><p>Evaluate the performance of the trained Random Forest model for sleep disorder prediction.</p></div>', unsafe_allow_html=True)

    # Display accuracy in a card-like format
    st.markdown('<div class="subheader">Model Accuracy</div>', unsafe_allow_html=True)
    st.metric(label="Accuracy", value=f"{accuracy:.2f}%")

    # Display confusion matrix
    st.markdown('<div class="subheader">Confusion Matrix</div>', unsafe_allow_html=True)
    st.write("Confusion Matrix from the model's evaluation on the test set:")
    st.image("models/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

    # Display classification report in tabular form
    st.markdown('<div class="subheader">Classification Report</div>', unsafe_allow_html=True)
    st.write("Classification Report with metrics for each class:")
    
    # Parse classification report from text
    lines = class_report.strip().split('\n')
    data = []
    headers = ['precision', 'recall', 'f1-score', 'support']
    class_names = label_encoder.classes_  # ['No Disease', 'Insomnia', 'Sleep Apnea']
    
    # Extract data rows
    for line in lines[2:]:  # Skip header lines
        if line.strip() and not line.startswith('accuracy') and not line.startswith('macro avg') and not line.startswith('weighted avg'):
            parts = re.split(r'\s+', line.strip())[1:5]  # Extract metrics
            data.append(parts)
        elif line.startswith('accuracy'):
            parts = re.split(r'\s+', line.strip())
            data.append(['', '', '', parts[-1]])  # Accuracy row
        elif line.startswith('macro avg') or line.startswith('weighted avg'):
            parts = re.split(r'\s+', line.strip())[2:6]  # Macro/weighted avg
            data.append(parts)
    
    # Create DataFrame
    index = list(class_names) + ['accuracy', 'macro avg', 'weighted avg']
    report_df = pd.DataFrame(data, columns=headers, index=index)
    report_df = report_df.astype(float, errors='ignore').round(4)
    
    st.dataframe(report_df, use_container_width=True)