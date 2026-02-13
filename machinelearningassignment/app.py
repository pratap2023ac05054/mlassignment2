import os
import streamlit as st

from src.data_loader import load_dataset, get_default_path
from src.models import train_and_evaluate


# -------------------------------
# Configuration
# -------------------------------
TARGET_COLUMN = "Price_Category"
TEST_SIZE = 0.2


# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.title("Machine Learning")
st.title("Assignment - 2")

st.title("Car Dataset Model Evaluation Dashboard")


# -------------------------------
# Load Dataset
# -------------------------------
data_path = get_default_path()

if not os.path.exists(data_path):
    st.error("Dataset not found. Please place 'global_cars_enhanced.csv' inside the data folder.")
    st.stop()

data = load_dataset(data_path)

# Check target column
if TARGET_COLUMN not in data.columns:
    st.error(f"Column '{TARGET_COLUMN}' not found in dataset.")
    st.stop()


# -------------------------------
# Display Basic Info
# -------------------------------
st.write(f"**Target column:** {TARGET_COLUMN}")
st.write(f"**Test size:** {TEST_SIZE}")


# -------------------------------
# Train Models
# -------------------------------
if st.button("Run Model Comparison"):

    with st.spinner("Training models... This may take a few seconds."):

        results_df, class_count = train_and_evaluate(
            data,
            target_col=TARGET_COLUMN,
            test_size=TEST_SIZE,
            include_xgb=True
        )

    st.success(f"Training completed. Number of classes: {class_count}")

    st.subheader("Model Performance")
    st.dataframe(results_df, use_container_width=True)