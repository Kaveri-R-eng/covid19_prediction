import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st

# Load and inspect the dataset
df = pd.read_csv("covid-19 symptoms dataset.csv")

# Check for missing values
is_null = df.isna().sum()
print("Null values in dataset:", is_null)

# Fill missing values with 0 or other meaningful values
df.fillna(0, inplace=True)

# Features and target
X = df.drop("infectionProb", axis=1)
y = df["infectionProb"]

# Ensure data is in correct format
X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)

# Check shape of X_train
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Clear previous Keras backend sessions
from tensorflow.keras import backend as K
K.clear_session()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure input data is of correct shape
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

# Define the model
model = Sequential([
    Dense(8, input_dim=X_train.shape[1], activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification (0 or 1)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Streamlit Web App
st.title("COVID-19 Detection Web App")
st.write("Enter the symptoms to predict if a person has COVID-19:")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Inputs for prediction
fever = st.selectbox("Fever (1 = Yes, 0 = No)", [0, 1])
cough = st.selectbox("Cough (1 = Yes, 0 = No)", [0, 1])
difficulty_breathing = st.selectbox("Difficulty Breathing (1 = Yes, 0 = No)", [0, 1])

# Ensure input features match training data
input_features = np.zeros((1, X_train.shape[1]))  # Create array with correct number of features
input_features[0, :3] = [fever, cough, difficulty_breathing]  # Fill relevant indices

# Predict COVID-19 status
if st.button("Predict COVID-19 Status"):
    prediction = model.predict(input_features, verbose=0)
    prediction_class = "COVID-19 Positive" if prediction[0][0] > 0.5 else "COVID-19 Negative"
    st.write(f"The model predicts: {prediction_class}")
