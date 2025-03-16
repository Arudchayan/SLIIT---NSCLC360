import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
import shap
import lime
import lime.lime_tabular

# Load the saved Cox model with caching
@st.cache_resource(ttl=3600)  # Refresh cache every hour
def load_model():
    return joblib.load("cox_model.pkl")

# Load historical patient data (Ensure this file exists)
def load_historical_data():
    try:
        return pd.read_csv("historical_patient_data.csv")
    except FileNotFoundError:
        st.warning("Historical patient data not found. Upload 'historical_patient_data.csv' to enable risk comparison.")
        return None

# Initialize the app
cox_model = load_model()
patients_data = load_historical_data()

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select a page",
    options=["Prognosis", "Detection", "Complications", "Recurrence"]
)

# Prognosis Page
if page == "Prognosis":
    st.title("Survival Prediction App")
    st.write("Upload a single row of patient data to get a survival curve.")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with one row", type=["csv"])

    if uploaded_file:
        # Read the uploaded CSV
        input_data = pd.read_csv(uploaded_file)

        # Ensure only one row
        if input_data.shape[0] != 1:
            st.error("Please upload exactly one row of data.")
        else:
            # Validate if the uploaded file contains required features
            expected_columns = cox_model.params_.index.tolist()  # Get model feature names
            missing_cols = [col for col in expected_columns if col not in input_data.columns]

            if missing_cols:
                st.error(f"Uploaded file is missing required columns: {', '.join(missing_cols)}")
            else:
                try:
                    # Predict survival function
                    survival_function = cox_model.predict_survival_function(input_data)

                    # Plot survival curve
                    fig, ax = plt.subplots()
                    survival_function.plot(ax=ax, color="blue", linewidth=2)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.set_title("Predicted Survival Curve", fontsize=14, fontweight="bold")
                    ax.set_xlabel("Time (months)", fontsize=12)
                    ax.set_ylabel("Survival Probability", fontsize=12)
                    st.pyplot(fig)

                    # Compute and display Concordance Index (C-index)
                    c_index = cox_model.concordance_index_
                    st.subheader(f"Model Concordance Index (C-index): {c_index:.3f}")

                    # Feature Importance (Top 10 absolute coefficients)
                    coef_df = cox_model.params_.sort_values(ascending=False)
                    top_features = coef_df.abs().sort_values(ascending=False).head(10)  # Top 10 influential genes

                    st.subheader("Top 10 Most Important Features")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=top_features.values, y=top_features.index, palette="coolwarm", ax=ax)
                    ax.set_title("Top 10 Important Features (Absolute Coefficients)")
                    ax.set_xlabel("Coefficient Magnitude")
                    ax.set_ylabel("Feature Name")
                    st.pyplot(fig)

                    # Compute Individual Risk Score
                    risk_score = (input_data * cox_model.params_).sum(axis=1).values[0]
                    st.subheader(f"Patient-Specific Risk Score: {risk_score:.3f}")

                    # Categorize Patient Risk using pd.cut to avoid bin label issues
                    # Define the bins based on the risk score range dynamically
                    min_risk_score = min(0, risk_score)  # Ensure no negative values
                    max_risk_score = max(1, risk_score)  # Ensure no risk score exceeds 1
                    bin_edges = [min_risk_score, 0.33, 0.66, max_risk_score]

                    # Categorize Risk based on these dynamic bins
                    risk_category = pd.cut([risk_score], bins=bin_edges, labels=["Low", "Medium", "High"], include_lowest=True)[0]
                    st.subheader(f"Risk Category: {risk_category} Risk")

                    # Load historical patient data for comparison (if available)
                    if patients_data is not None:
                        historical_risk_scores = (patients_data * cox_model.params_).sum(axis=1)

                        # Plot Risk Score Distribution
                        st.subheader("Risk Score Comparison with Historical Patients")
                        fig, ax = plt.subplots()
                        sns.histplot(historical_risk_scores, bins=30, kde=True, color="gray")
                        ax.axvline(risk_score, color="red", linestyle="dashed", linewidth=2, label="This Patient")
                        ax.set_title("Risk Score Distribution")
                        ax.set_xlabel("Risk Score")
                        ax.set_ylabel("Frequency")
                        ax.legend()
                        st.pyplot(fig)

                    # SHAP Explanation
                    explainer = shap.Explainer(cox_model)  # SHAP Explainer
                    shap_values = explainer(input_data)  # Calculate SHAP values for the input data
                    st.subheader("SHAP Summary Plot")
                    shap.summary_plot(shap_values, input_data)  # Visualize SHAP values for all features

                    # LIME Explanation
                    def predict_survival_function_lime(input_data):
                        input_data = input_data.values
                        return cox_model.predict_survival_function(input_data).T  # Transpose to match LIME's expected format

                    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                        training_data=patients_data.values,  # Use historical data to train the explainer
                        mode="regression",  # We are explaining a regression model (Cox PH)
                        feature_names=cox_model.params_.index.tolist(),
                        discretize_continuous=True
                    )

                    lime_explanation = explainer_lime.explain_instance(
                        input_data.values[0],  # The first (and only) row in the uploaded data
                        predict_survival_function_lime
                    )
                    st.subheader("LIME Explanation")
                    lime_explanation.show_in_notebook()  # Show the explanation in the notebook or Streamlit

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Detection Page
elif page == "Detection":
    st.title("Detection")
    st.write("Content related to disease detection goes here.")

# Complications Page
elif page == "Complications":
    st.title("Complications")
    st.write("Content related to complications goes here.")

# Recurrence Page
elif page == "Recurrence":
    st.title("Recurrence")
    st.write("Content related to recurrence goes here.")
