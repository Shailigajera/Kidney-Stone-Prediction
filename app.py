import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Set the title and background image
st.title("Kidney Stone Prediction")
st.markdown(
    """
    <style>
    body {
        background-image: url("https://www.kidneystonemedicine.com/wp-content/uploads/2020/02/kidney-stone.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the dataset
df = pd.read_csv("Data/Kidney Stone Detection.csv")

# Split the data into features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rfc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rfc.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Create a function to make predictions on new data
def make_prediction(data):
    prediction = rfc.predict_proba(data)
    return prediction

# Create a form to input new data
st.header("Input New Data")
with st.form("input_form"):
    gravity = st.number_input("Gravity", min_value=1.0, max_value=1.1, step=0.01)
    ph = st.number_input("pH", min_value=4.0, max_value=8.0, step=0.1)
    osmolality = st.number_input("Osmolality", min_value=200, max_value=1200, step=10)
    conductivity = st.number_input("Conductivity", min_value=10, max_value=40, step=1)
    urea = st.number_input("Urea", min_value=100, max_value=500, step=10)
    calcium = st.number_input("Calcium", min_value=1.0, max_value=10.0, step=0.1)
    submit_button = st.form_submit_button("Submit")

# Make predictions on new data
if submit_button:
    new_data = pd.DataFrame(
        {
            "gravity": [gravity],
            "ph": [ph],
            "osmolality": [osmolality],
            "conductivity": [conductivity],
            "urea": [urea],
            "calcium": [calcium],
        }
    )
    prediction = make_prediction(new_data)
    probability = prediction[0][1]
    st.write(f"You have a {probability*100:.2f}% chance of having a kidney stone.")
