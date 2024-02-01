#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pickle

# Load your dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv('diabetes.csv')

# Perform data preprocessing
# Fill missing, NaN, and null values with the mean for all columns
imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Features and target variable
X = data_filled[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data_filled['Outcome']  # Assuming 'Outcome' is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Save the trained model as a pickle file
model_path = 'random_forest_model.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(random_forest_classifier, model_file)

# Streamlit app
st.title("Diabetes Prediction App")

# User input form
st.sidebar.header("Enter Patient Information")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=0, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=0, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=0, step=1)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=500, value=0, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
diabetes_pedigree_function = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=0, step=1)

# Button to trigger prediction
predict_button = st.sidebar.button("Predict")

if predict_button:
    # Create a DataFrame with the user input
    user_input = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    # Make prediction
    prediction = random_forest_classifier.predict(user_input)

    # Display prediction
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("You are diabetic.")
    else:
        st.write("Congratulations! You are not diabetic.")


# In[ ]:




