import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title
st.title("Linear Regression App")

# Sidebar for user inputs
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Feature selection
    st.sidebar.header("Feature Selection")
    features = st.sidebar.multiselect("Select Features (X)", data.columns)
    target = st.sidebar.selectbox("Select Target (y)", data.columns)

    if features and target:
        # Split the data
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Display model performance
        st.write("### Model Performance")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

        # Visualization
        st.write("### Predictions vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Show coefficients
        st.write("### Model Coefficients")
        coefficients = pd.DataFrame(model.coef_, features, columns=["Coefficient"])
        st.write(coefficients)
