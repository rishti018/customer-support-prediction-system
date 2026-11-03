import streamlit as st
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎫"
)

# Model paths
CATEGORY_MODEL_PATH = "models/category_model.pkl"
PRIORITY_MODEL_PATH = "models/priority_model.pkl"

# Load trained models
@st.cache_resource
def load_models():

    with open(CATEGORY_MODEL_PATH, "rb") as file:
        category_model = pickle.load(file)

    with open(PRIORITY_MODEL_PATH, "rb") as file:
        priority_model = pickle.load(file)

    return category_model, priority_model

# Check whether models exist
if not os.path.exists(CATEGORY_MODEL_PATH) or not os.path.exists(PRIORITY_MODEL_PATH):

    st.error("Trained models not found. Please run train.py first.")
    st.stop()

# Load models
try:
    category_model, priority_model = load_models()

except Exception as error:
    st.error(f"Error loading models: {error}")
    st.stop()

# Application heading
st.markdown(
    "<h2 style='font-size: 32px; font-weight: 600;'>"
    "Customer Support Ticket Classification</h2>",
    unsafe_allow_html=True
)
st.write(
    "Predict the category and priority of customer support tickets using Machine Learning."
)

st.divider()

# Ticket input
subject = st.text_input("Ticket Subject")

body = st.text_area(
    "Ticket Description",
    height=180
)

# Prediction
if st.button("Predict Ticket"):

    if not subject.strip() or not body.strip():

        st.warning("Please enter both the subject and description.")

    else:

        ticket_text = subject.strip() + " " + body.strip()

        try:
            category = category_model.predict([ticket_text])[0]
            priority = priority_model.predict([ticket_text])[0]

            st.success("Prediction completed successfully!")

            st.subheader("Prediction Results")

            st.write("**Predicted Category:**", category)
            st.write("**Predicted Priority:**", priority)

        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.divider()

st.caption("Machine Learning Project | TF-IDF and Linear SVM")