import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration and Setup ---
# Set the page title and icon
st.set_page_config(page_title="Student Performance Predictor", page_icon="📊")

# --- Load Model and Preprocessor ---
# It's good practice to load these once when the app starts,
# rather than every time a user interacts with a widget.
# Streamlit's caching mechanism (st.cache_resource) is perfect for this.
@st.cache_resource
def load_assets():
    """
    Loads the preprocessor and the trained machine learning model.
    These files ('preprocessor.pkl', 'linear_regression_model.pkl')
    are expected to be in the same directory as this app.py script.
    """
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        model = joblib.load('linear_regression_model.pkl')
        return preprocessor, model
    except FileNotFoundError:
        st.error("Error: Model or preprocessor files not found. "
                 "Please ensure 'preprocessor.pkl' and 'linear_regression_model.pkl' "
                 "are in the same directory as app.py.")
        st.stop() # Stop the app execution if files are missing
    except Exception as e:
        st.error(f"An error occurred while loading assets: {e}")
        st.stop()

# Load the assets
preprocessor, model = load_assets()

# --- Streamlit UI Elements ---
st.title('📊 Student Performance Indicator') # Updated title
st.markdown("""
    This application helps indicate a student's potential math score based on various
    demographic and academic factors.
    Fill in the details below and click 'Predict' to see the estimated performance.
""") # Updated description

# Input widgets for features
# Categorical features - using selectbox for predefined options
st.header("Student Information")

gender = st.selectbox(
    'Gender',
    ['female', 'male'],
    help="Select the student's gender."
)

race_ethnicity = st.selectbox(
    'Race/Ethnicity Group',
    ['group A', 'group B', 'group C', 'group D', 'group E'],
    help="Select the student's race/ethnicity group."
)

parental_level_of_education = st.selectbox(
    'Parental Level of Education',
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
    help="Select the highest education level of the student's parents."
)

lunch = st.selectbox(
    'Lunch Type',
    ['standard', 'free/reduced'],
    help="Select the type of lunch the student receives."
)

test_preparation_course = st.selectbox(
    'Test Preparation Course',
    ['none', 'completed'],
    help="Indicate if the student completed a test preparation course."
)

# Numerical features - using sliders for a range of values
reading_score = st.slider(
    'Reading Score (0-100)',
    min_value=0, max_value=100, value=70,
    help="Enter the student's reading score."
)

writing_score = st.slider(
    'Writing Score (0-100)',
    min_value=0, max_value=100, value=70,
    help="Enter the student's writing score."
)

# --- Prediction Logic ---
if st.button('Predict Math Score'):
    # Create a DataFrame from the user's input
    # The column names must exactly match those used during model training
    input_data = pd.DataFrame([[
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score
    ]], columns=[
        'gender',
        'race_ethnicity',
        'parental_level_of_education',
        'lunch',
        'test_preparation_course',
        'reading_score',
        'writing_score'
    ])

    try:
        # Apply the preprocessor to transform the input data
        # This step is crucial for consistent feature scaling and one-hot encoding
        processed_input_data = preprocessor.transform(input_data)

        # Make the prediction using the loaded model
        predicted_math_score = model.predict(processed_input_data)[0]

        # Display the prediction to the user
        st.success(f'Predicted Math Score: **{predicted_math_score:.2f}**')
        st.info("Note: This is a prediction based on the trained model and provided inputs.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your inputs and try again.")

st.markdown("---")
st.markdown("Developed by Adi – Student Performance Predictor 🚀")

