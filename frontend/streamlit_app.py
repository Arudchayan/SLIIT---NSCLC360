import streamlit as st
import requests

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select a page",
    options=["Prognosis", "Detection", "Complications", "Recurrence"]
)

# Base URL for Flask backend
BACKEND_URL = "http://localhost:5000"

# Minimal frontend: upload file and send to backend for predictions
if page == "Prognosis":
    st.title("Survival Analysis")
    st.write("Upload a CSV file with one row of patient data to get predictions from the backend.")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded_file:
        # Send file to Flask backend
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        try:
            response = requests.post(f"{BACKEND_URL}/prognosis", files=files)
            if response.ok:
                data = response.json()
                # Display returned elements
                if "plot" in data:
                    st.image(data["plot"], caption="Predicted Survival Curve")
                if "c_index" in data:
                    st.write(f"Model Concordance Index (C-index): {data['c_index']:.3f}")
                if "risk_category" in data:
                    st.write(f"Risk Category: {data['risk_category']}")
                # Additional fields can be displayed as needed
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# Stub pages
elif page == "Detection":
    st.title("Detection")
    st.write("Upload data and receive detection results from the backend.")
    uploaded_file = st.file_uploader("Choose file", type=["csv"] )
    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        try:
            response = requests.post(f"{BACKEND_URL}/detection", files=files)
            if response.ok:
                st.json(response.json())
            else:
                st.error(f"Detection failed: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

elif page == "Complications":
    st.title("Complications")
    st.write("Complications analysis coming from backend.")
    # Similar pattern can be implemented here
    
elif page == "Recurrence":
    st.title("Recurrence")
    st.write("Recurrence predictions coming from backend.")
    # Similar pattern can be implemented here
