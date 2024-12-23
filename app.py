import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title and description
st.title("Linear Regression Application")
st.write("This application allows you to perform linear regression on a dataset.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Read the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Select features and target
    st.write("Select the features (X) and target (y):")
    columns = data.columns.tolist()
    features = st.multiselect("Select features (X):", columns)
    target = st.selectbox("Select target (y):", columns)

    if features and target:
        # Split the data
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Display metrics
        st.write("### Model Performance")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

        # Visualization
        st.write("### Actual vs Predicted")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        st.pyplot(plt)

        # Display coefficients
        st.write("### Model Coefficients")
        coeff_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
        st.write(coeff_df)\
