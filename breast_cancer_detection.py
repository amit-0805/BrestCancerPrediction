import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Breast Cancer Detection")

# Load the dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv("breast cancer kaggle.csv")
    dataset = dataset.drop(columns='Unnamed: 32')
    return dataset

dataset = load_data()

# ... (rest of the code remains the same)

if st.sidebar.checkbox("Show raw data"):
    st.write(dataset)

# Data Preprocessing
st.header("Data Preprocessing")
if st.sidebar.checkbox("Show dataset info"):
    st.write(dataset.info())

if st.sidebar.checkbox("Show dataset description"):
    st.write(dataset.describe())

if st.sidebar.checkbox("Show missing values"):
    st.write(dataset.isnull().sum())

if st.sidebar.checkbox("Show correlation matrix"):
    corr = dataset.corr()
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

# Feature Selection

dataset = pd.get_dummies(data=dataset, drop_first=True)

# Train-Test Split
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Model Training
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Model Prediction
y_pred = classifier.predict(x_test)

# Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.header("Model Evaluation")
st.write(f"Accuracy: {acc:.2f}")
st.write(f"Precision: {prec:.2f}")
st.write(f"Recall: {rec:.2f}")
st.write(f"F1 Score: {f1:.2f}")

if st.sidebar.checkbox("Show prediction results"):
    st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

# New prediction form
st.header("Predict Breast Cancer")
st.write("Enter the values for prediction:")

# Create a form
with st.form("prediction_form"):
    # Add input fields for each feature
    # Adjust these based on your actual features
    radius_mean = st.number_input("Radius Mean")
    texture_mean = st.number_input("Texture Mean")
    perimeter_mean = st.number_input("Perimeter Mean")
    area_mean = st.number_input("Area Mean")
    smoothness_mean = st.number_input("Smoothness Mean")
    compactness_mean = st.number_input("Compactness Mean")
    concavity_mean = st.number_input("Concavity Mean")
    concave_points_mean = st.number_input("Concave Points Mean")
    symmetry_mean = st.number_input("Symmetry Mean")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean")
    radius_se = st.number_input("Radius SE")
    texture_se = st.number_input("Texture SE")
    perimeter_se = st.number_input("Perimeter SE")
    area_se = st.number_input("Area SE")
    smoothness_se = st.number_input("Smoothness SE")
    compactness_se = st.number_input("Compactness SE")
    concavity_se = st.number_input("Concavity SE")
    concave_points_se = st.number_input("Concave Points SE")
    symmetry_se = st.number_input("Symmetry SE")
    fractal_dimension_se = st.number_input("Fractal Dimension SE")
    radius_worst = st.number_input("Radius Worst")
    texture_worst = st.number_input("Texture Worst")
    perimeter_worst = st.number_input("Perimeter Worst")
    area_worst = st.number_input("Area Worst")
    smoothness_worst = st.number_input("Smoothness Worst")
    compactness_worst = st.number_input("Compactness Worst")
    concavity_worst = st.number_input("Concavity Worst")
    concave_points_worst = st.number_input("Concave Points Worst")
    symmetry_worst = st.number_input("Symmetry Worst")
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst")

    # Submit button
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare the input data
        input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                                fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                                smoothness_se, compactness_se, concavity_se, concave_points_se,
                                symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                                perimeter_worst, area_worst, smoothness_worst, compactness_worst,
                                concavity_worst, concave_points_worst, symmetry_worst,
                                fractal_dimension_worst]])

        # Scale the input data
        input_scaled = sc.transform(input_data)

        # Make prediction
        prediction = classifier.predict(input_scaled)

        # Display result
        if prediction[0] == 1:
            st.write("The model predicts: Malignant")
        else:
            st.write("The model predicts: Benign")

        # Display prediction probability
        proba = classifier.predict_proba(input_scaled)[0]
        st.write(f"Probability of being benign: {proba[0]:.2f}")
        st.write(f"Probability of being malignant: {proba[1]:.2f}")