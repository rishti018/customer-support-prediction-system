import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎫"
)

API_URL = os.getenv(
    "API_URL",
    "https://customer-support-prediction-system.onrender.com/predict"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
h2 {
    margin-top: 0rem;
    padding-top: 0rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h2 style='font-size: 32px; font-weight: 600;'>"
    "Customer Support Ticket Classification</h2>",
    unsafe_allow_html=True
)

st.write(
    "Predict the category and priority of customer support tickets using Machine Learning."
)

st.divider()

subject = st.text_input("Ticket Subject")
body = st.text_area("Ticket Description", height=180)

if st.button("Predict Ticket"):

    if not subject.strip() or not body.strip():
        st.warning("Please enter both the subject and description.")

    else:
        try:
            response = requests.post(
                API_URL,
                json={
                    "subject": subject,
                    "body": body,
                    "model": "svm"
                },
                timeout=60
            )

            if response.ok:
                result = response.json()

                st.success("Prediction completed successfully!")
                st.subheader("Prediction Results")
                st.write("**Predicted Category:**", result["category"])
                st.write("**Predicted Priority:**", result["priority"])

            else:
                st.error(response.json().get("error", "API request failed"))

        except requests.RequestException as error:
            st.error(f"Could not connect to prediction API: {error}")

st.divider()
st.caption("Machine Learning Project | TF-IDF and Linear SVM")