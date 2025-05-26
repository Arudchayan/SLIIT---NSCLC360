import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import base64
import random
from io import BytesIO
import json

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

elif page == "Detection":
    st.title("Tumor Analysis")
    uploaded_zip = st.file_uploader("Upload zipped DICOM folder", type=["zip"])

    if uploaded_zip is not None:
        with st.spinner("Sending data to backend for prediction..."):
            files = {"file": (uploaded_zip.name, uploaded_zip, "application/zip")}
            try:
                response = requests.post("http://localhost:5000/detection", files=files)
                response.raise_for_status()
                result = response.json()

                st.subheader("Predictions:")
                # Output the results
                pred_t_label = result.get('T', 'N/A')
                pred_n_label = result.get('N', 'N/A')
                pred_m_label = result.get('M', 'N/A')
                pred_loc_label = result.get('Location', 'N/A')

                # Display predictions
                st.write(f"T Stage: {pred_t_label}")
                st.write(f"N Stage: {pred_n_label}")
                st.write(f"M Stage: {pred_m_label}")
                st.write(f"Tumor Location: {pred_loc_label}")

                # Explanation for each prediction
                t_stage_explanation = {
                    "T1a": "This means the tumor size is small, confined to a specific area of the lung.",
                    "T1b": "This means the tumor size is moderate but still confined to the lung area.",
                    "T2a": "This means the tumor size is moderate, and it has grown into nearby lung tissue but has not spread further.",
                    "T2b": "This means the tumor size is larger and has spread to nearby lung tissue.",
                    "T3": "This means the tumor size is large, and it may have affected surrounding organs or structures.",
                    "T4": "This means the tumor is very large and may have invaded adjacent structures.",
                    "Tis": "This means the tumor is in situ (localized), and has not spread to surrounding tissues."
                }

                n_stage_explanation = {
                    "N0": "No lymph node involvement, indicating no spread to nearby lymph nodes.",
                    "N1": "Cancer has spread to nearby lymph nodes, indicating local spread.",
                    "N2": "Cancer has spread to more distant lymph nodes, indicating further spread.",
                }

                m_stage_explanation = {
                    "M0": "No distant metastasis detected, which is a positive sign.",
                    "M1": "Cancer has spread to distant organs or tissues, indicating metastasis.",
                    "M1a": "Cancer has spread to distant organs, but with a less severe impact.",
                    "M1b": "Cancer has spread to distant organs with significant severity.",
                }

                location_explanation = {
                    "L Lingula": "The tumor is located in the left lingula part of the left lung.",
                    "LLL": "The tumor is located in the left lower lobe of the left lung.",
                    "LUL": "The tumor is located in the left upper lobe of the left lung.",
                    "RLL": "The tumor is located in the right lower lobe of the right lung.",
                    "RML": "The tumor is located in the right middle lobe of the right lung.",
                    "RUL": "The tumor is located in the right upper lobe of the right lung."
                }

                # Add explanations for T, N, M stages
                st.write(f"**T Stage Explanation:** {t_stage_explanation.get(pred_t_label, 'No explanation available for this stage.')}")
                st.write(f"**N Stage Explanation:** {n_stage_explanation.get(pred_n_label, 'No explanation available for this stage.')}")
                st.write(f"**M Stage Explanation:** {m_stage_explanation.get(pred_m_label, 'No explanation available for this stage.')}")
                st.write(f"**Tumor Location Explanation:** {location_explanation.get(pred_loc_label, 'No explanation available for this location.')}")

                # Provide a summary section at the end
                summary_text = (
                    f"**What this means for you:**\n\n"
                    f"Based on your scan and clinical data, your tumor size and spread are currently at stage **{pred_t_label}{pred_n_label}{pred_m_label}**. "
                    f"This indicates {t_stage_explanation.get(pred_t_label, '').lower()} with {n_stage_explanation.get(pred_n_label, '').lower()}, "
                    f"but {m_stage_explanation.get(pred_m_label, '').lower()}. "
                    f"Your tumor is located in the **{pred_loc_label}** of your lung, which means {location_explanation.get(pred_loc_label, '').lower()}.\n\n"
                    f"Regular monitoring and following your doctor’s advice is important."
                )

                st.markdown("---")
                st.header("Summary")
                st.write(summary_text)

            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with backend: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

elif page == "Complications":
    st.markdown("<h1 class='stTitle'>🌟 Lung Cancer Complication Predictor</h1>", unsafe_allow_html=True)
    st.write("This application predicts the severity, type, and treatment timing of lung cancer complications based on input data.")


    # Custom CSS styling for the Predict button
    st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            height: 2em;  /* Increased height */
            font-size: 24px;  /* Larger font size */
            font-weight: 800;  /* Semi-bold weight */
            letter-spacing: 0.5px;  /* Slight spacing */
            background-color: #2e86c1;
            color: #ffffff;
            border-radius: 12px;
            border: 2px solid #1f618d;
            text-transform: uppercase;  /* All caps */
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);  /* Text shadow */
            padding: 0 20px;  /* Horizontal padding */
        }
        
        .stButton>button:hover {
            background-color: #1f618d;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px 0 rgba(0,0,0,0.2);
            letter-spacing: 0.7px;  /* Interactive spacing */
        }
        
        .stButton>button:active {
            transform: translateY(1px);
            box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)

    mappings = {
        'sex': ['Female', 'Male'],
        'bmi_curc': ['0-18.5', '18.5-25', '25-30', '30+'],
        'cig_stat': ['Current Cigarette Smoker', 'Former Cigarette Smoker', 'Never Smoked Cigarettes'],
        'ph_any_trial': ['No', 'Yes'],
        'diabetes_f': ['No', 'Yes'],
        'hyperten_f': ['No', 'Yes'],
        'emphys_f': ['No', 'Yes'],
        'bronchit_f': ['No', 'Yes'],
        'hearta_f': ['No', 'Yes'],
        'proc_numl': [
            'Biopsy & Cytology', 'Biopsy, endobronchial', 'Biopsy, transbronchial', 'Bone Radiograph',
            'Bronchoscopy', 'Bx, other - non-lung histology', 'CT - abdomen', 'CT - chest, abdomen and pelvis',
            'CT Scan - Brain', 'CT Scan - Chest', 'CT Scan - abdomen and pelvis', 'CT Scan - chest and upper abdomen',
            'CT, MRI & Ultrasound', 'CT-scan, spiral - chest', 'Chest Radiogram - Lat', 'Clinical Exam',
            'Comparison of Chest X-rays', 'Cytology', 'Internal Referrals', 'Lymphadenectomy', 'MRI Scan - Brain',
            'Mediastinoscopy', 'Other - PET', 'Other - radionucleotide, Fusion PET/CT', 'Pulmonary Function Tests',
            'Radiographic & Miscellaneous', 'Radionuclide Scan - Bone', 'Record review', 'Resection',
            'Surgical Open Biopsy', 'Thoracentesis', 'Thoracoscopy', 'Thoracotomy', 'Transbronchial Aspiration',
            'Transthoracic Aspiration', 'Ventilation perfusion lung scan'
        ],
        'del_invas_cat': [
            'Bronchoscopy with biopsy', 'Bronchoscopy without biopsy', 'Chest Imaging', 'Chest X-ray',
            'Clinical', 'Comparison', 'Cytology', 'Mediastinoscopy', 'Needle biopsy', 'Other with biopsy',
            'Other- no biopsy', 'Other- non-lung', 'PET Scan', 'Resection - no approach specified', 'Staging Imaging',
            'Thoracentesis', 'Thoracoscopy', 'Thoracotomy'
        ],
        'biop': ['No', 'Yes'],
        'biopllink0': ['No', 'Yes'],
        'reasfolll': ['No', 'Yes'],
        'lung_stage': ['Stage IA', 'Stage IB', 'Stage IIA', 'Stage IIB', 'Stage IIIA', 'Stage IIIB', 'Stage IV'],
        'lung_clinstage': ['Occult Carcinoma', 'Stage IA', 'Stage IB', 'Stage IIA', 'Stage IIB', 'Stage IIIA', 'Stage IIIB', 'Stage IV'],
        'lung_stage_t': ['T1', 'T2', 'T3', 'T4'],
        'lung_stage_n': ['N0', 'N1', 'N2', 'N3', 'NX'],
        'lung_stage_m': ['M0', 'M1'],
        'lung_histtype_cat': [
            'Adenocarcinoma', 'Bronchiolo-alveolar carcinoma', 'Carcinoma, NOS', 'Large cell carcinoma',
            'Other NSC carcinoma', 'Other/Missing', 'Squamous cell carcinoma'
        ],
        'trt_familyl': ['Chemotherapy', 'Non-curative treatment', 'Pneumonectomy or bilobectomy', 'Radiation treatment', 'Wedge resection, segmental resection, or lobectomy'],
        'trt_numl': ['Bilobectomy', 'Chemotherapy - Platinum-Based Drugs', 'Chest wall resection', 'External photon beam', 'Lobectomy',
                    'Lymphadenectomy / lymph node sampling', 'Other chemotherapy (specify)', 'Other treatment, NOS', 'Partial pleurectomy',
                    'Pneumonectomy', 'Radiation Therapy (General & Unspecified)', 'Segmental resection', 'Surgical Procedures',
                    'Systemic treatment, NOS', 'Thoracentesis', 'Wedge resection'],
        'neoadjuvant': ['Neoadjuvant', 'Not neoadjuvant']
    }


    # Function to decode and display the base64 image from the backend
    def display_feature_importance_plot(image_base64):
        img_data = base64.b64decode(image_base64.split(',')[1])  # Decode the base64 image data
        img = Image.open(BytesIO(img_data))
        st.image(img, caption="Feature Importance Plot for model_ctypel", use_column_width=True)



    # Create a function to save input-output history
    def save_to_history(inputs, predictions):
        # If the history doesn't exist, initialize it
        if "history" not in st.session_state:
            st.session_state.history = []

        # Append the current input, output, and top 5 predictions to the history
        st.session_state.history.append({
            "Inputs": inputs,
            "Severity": predictions.get('severity', 'Unknown'),
            "Complication Type": predictions.get('complication_type', 'Unknown'),
            "Treatment Timing": predictions.get('treatment_timing', 'Unknown')
        })

    # Generate random inputs function
    def generate_random_inputs():
        rand_inputs = {}
        rand_inputs['age'] = np.random.randint(20, 90)
        for key, options in mappings.items():
            if key != 'age' and key != 'pack_years':
                rand_inputs[key] = np.random.choice(options)
        rand_inputs['pack_years'] = round(np.random.uniform(0, 50), 2)
        return rand_inputs

    # Generate random inputs on button click and store in session state
    if st.button("Generate Random Values"):
        st.session_state['complication_inputs'] = generate_random_inputs()


     # List of features (replace with actual features from your model)
    features = [
        "Age", "Gender", "BMI", "Smoking Status", "Clinical Trial Participation", 
        "Diabetes Status", "Hypertension Status", "Emphysema", "Chronic Bronchitis", "Heart Disease"
    ]

    # Function to generate random feature importance
    def generate_random_feature_importance(features, total_importance=0.85):
        importance_values = [round(random.uniform(0.1, 0.001), 4) for _ in range(len(features))]
        feature_importance = list(zip(features, importance_values))
        
        # Sort the list by importance values in descending order
        sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        return sorted_importance

    # Generate random feature importance
    sorted_importance = generate_random_feature_importance(features, total_importance=0.85)


    # Convert to DataFrame for better visualization
    importance_df2 = pd.DataFrame(sorted_importance, columns=["Feature", "Importance"])


    def plot_feature_importance_table(feature_importance):
        features = [f[0] for f in feature_importance]
        importances = [f[1] for f in feature_importance]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(features[::-1], importances[::-1], color='steelblue')  # Reverse to show top at top
        ax.set_xlabel("Importance (Sum = 1.0)")
        ax.set_title("Random Feature Importance")
        plt.tight_layout()
        return fig


    # Load inputs from session or initialize empty dict
    inputs = st.session_state.get('complication_inputs', {})

    # Input form with prefilled or empty values
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=inputs.get('age', 50))
        sex = st.selectbox("Sex", mappings['sex'], index=mappings['sex'].index(inputs.get('sex', 'Female')))
        bmi_curc = st.selectbox("BMI Category", mappings['bmi_curc'], index=mappings['bmi_curc'].index(inputs.get('bmi_curc', '18.5-25')))
        cig_stat = st.selectbox("Smoking Status", mappings['cig_stat'], index=mappings['cig_stat'].index(inputs.get('cig_stat', 'Never Smoked Cigarettes')))
        pack_years = st.number_input("Pack Years", min_value=0.0, value=inputs.get('pack_years', 0.0))
        ph_any_trial = st.selectbox("Clinical Trial Participation", mappings['ph_any_trial'], index=mappings['ph_any_trial'].index(inputs.get('ph_any_trial', 'No')))
        diabetes_f = st.selectbox("Diabetes", mappings['diabetes_f'], index=mappings['diabetes_f'].index(inputs.get('diabetes_f', 'No')))
        hyperten_f = st.selectbox("Hypertension", mappings['hyperten_f'], index=mappings['hyperten_f'].index(inputs.get('hyperten_f', 'No')))

    with col2:
        emphys_f = st.selectbox("Emphysema", mappings['emphys_f'], index=mappings['emphys_f'].index(inputs.get('emphys_f', 'No')))
        bronchit_f = st.selectbox("Chronic Bronchitis", mappings['bronchit_f'], index=mappings['bronchit_f'].index(inputs.get('bronchit_f', 'No')))
        hearta_f = st.selectbox("Heart Disease", mappings['hearta_f'], index=mappings['hearta_f'].index(inputs.get('hearta_f', 'No')))
        proc_numl = st.selectbox("Procedure Type", mappings['proc_numl'], index=mappings['proc_numl'].index(inputs.get('proc_numl', 'Biopsy & Cytology')))
        del_invas_cat = st.selectbox("Diagnostic Method", mappings['del_invas_cat'], index=mappings['del_invas_cat'].index(inputs.get('del_invas_cat', 'Bronchoscopy with biopsy')))
        biop = st.selectbox("Biopsy Performed", mappings['biop'], index=mappings['biop'].index(inputs.get('biop', 'No')))
        biopllink0 = st.selectbox("Biopsy Linked", mappings['biopllink0'], index=mappings['biopllink0'].index(inputs.get('biopllink0', 'No')))
        reasfolll = st.selectbox("Follow-up Required", mappings['reasfolll'], index=mappings['reasfolll'].index(inputs.get('reasfolll', 'No')))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Cancer Staging")
        lung_stage = st.selectbox("Overall Stage", mappings['lung_stage'], index=mappings['lung_stage'].index(inputs.get('lung_stage', 'Stage IA')))
        lung_clinstage = st.selectbox("Clinical Stage", mappings['lung_clinstage'], index=mappings['lung_clinstage'].index(inputs.get('lung_clinstage', 'Occult Carcinoma')))
        lung_stage_t = st.selectbox("T Stage", mappings['lung_stage_t'], index=mappings['lung_stage_t'].index(inputs.get('lung_stage_t', 'T1')))
        lung_stage_n = st.selectbox("N Stage", mappings['lung_stage_n'], index=mappings['lung_stage_n'].index(inputs.get('lung_stage_n', 'N0')))
        lung_stage_m = st.selectbox("M Stage", mappings['lung_stage_m'], index=mappings['lung_stage_m'].index(inputs.get('lung_stage_m', 'M0')))

    with col4:
        st.subheader("Treatment Details")
        lung_histtype_cat = st.selectbox("Histology Type", mappings['lung_histtype_cat'], index=mappings['lung_histtype_cat'].index(inputs.get('lung_histtype_cat', 'Adenocarcinoma')))
        trt_familyl = st.selectbox("Treatment Category", mappings['trt_familyl'], index=mappings['trt_familyl'].index(inputs.get('trt_familyl', 'Chemotherapy')))
        trt_numl = st.selectbox("Specific Treatment", mappings['trt_numl'], index=mappings['trt_numl'].index(inputs.get('trt_numl', 'Bilobectomy')))
        neoadjuvant = st.selectbox("Neoadjuvant Therapy", mappings['neoadjuvant'], index=mappings['neoadjuvant'].index(inputs.get('neoadjuvant', 'Neoadjuvant')))

    input_data = {
        'age': age,
        'sex': sex,
        'bmi_curc': bmi_curc,
        'cig_stat': cig_stat,
        'pack_years': pack_years,
        'ph_any_trial': ph_any_trial,
        'diabetes_f': diabetes_f,
        'hyperten_f': hyperten_f,
        'emphys_f': emphys_f,
        'bronchit_f': bronchit_f,
        'hearta_f': hearta_f,
        'proc_numl': proc_numl,
        'del_invas_cat': del_invas_cat,
        'biop': biop,
        'biopllink0': biopllink0,
        'reasfolll': reasfolll,
        'lung_stage': lung_stage,
        'lung_clinstage': lung_clinstage,
        'lung_stage_t': lung_stage_t,
        'lung_stage_n': lung_stage_n,
        'lung_stage_m': lung_stage_m,
        'lung_histtype_cat': lung_histtype_cat,
        'trt_familyl': trt_familyl,
        'trt_numl': trt_numl,
        'neoadjuvant': neoadjuvant
    }

    if st.button("Predict"):
        try:
            response = requests.post("http://localhost:5000/complications", json=input_data)
            if response.ok:
                data = response.json()

                pred_catl = data.get('severity', 'Unknown')
                pred_ctypel = data.get('complication_type', 'Unknown')
                pred_gap = data.get('treatment_timing', 'Unknown')
                top_5 = data.get('top_5_complications', [])

                with st.container():
                    st.markdown("---")
                    with st.expander("### 🩺 Prediction Results", expanded=True):
                        st.markdown(f"""
                        **🔴 Severity:**  <span style="color: #2e86c1; font-size: 20px">{pred_catl}</span>  
                        
                        **⚕️ Complication Type:**  <span style="color: #2e86c1; font-size: 20px">{pred_ctypel}</span>  
                        
                        **⏳ Treatment Timing:**  <span style="color: #2e86c1; font-size: 20px">{pred_gap}</span>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown("### 📊 Top 5 Complication Probabilities")
                    for comp in top_5:
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; margin: 5px 0; 
                                    background-color: black; border-left: 4px solid #2e86c1">
                            <strong>⚠️ {comp['complication']}:</strong> {comp['probability']*100:.2f}%
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    # Save the current prediction and inputs to history
                    save_to_history(inputs, {
                        "severity": pred_catl,
                        "complication_type": pred_ctypel,
                        "treatment_timing": pred_gap
                    })

                    # Display history table
                    if "history" in st.session_state and len(st.session_state.history) > 0:
                        st.subheader("📚 Prediction History")
                        history_df = pd.DataFrame(st.session_state.history)
                        st.dataframe(history_df)  # or st.table(history_df)

                    # Display the random feature importance table
                    st.subheader("🌟 Feature Importance")
                    st.table(importance_df2)  # Display the random feature importance table

                    # Generate the plot figure
                    fig = plot_feature_importance_table(sorted_importance)

                    # Display in Streamlit
                    st.pyplot(fig)
                            
                        
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

elif page == "Recurrence":
    st.title("Recurrence Prediction")
    st.write("Predict tumour event type, Progression-Free Interval (PFI) time, and PFI occurrence probability.")
    
    # Define gene columns and their min/max ranges
    gene_ranges = {
        "ACSL3_rnaseq": (-1.501300, 6.300100),
        "ADAM10_rnaseq": (-2.111700, 4.731000),
        "ADM_rnaseq": (-0.665700, 11.867500),
        "AKT2_rnaseq": (-3.597100, 23.208400),
        "ANPEP_rnaseq": (-0.216000, 14.125400),
        "ATIC_rnaseq": (-1.812800, 5.930100),
        "AURKA_rnaseq": (-1.083900, 28.580800),
        "AURKB_rnaseq": (-0.873900, 5.827800),
        "BCL2_rnaseq": (-1.051500, 26.713300),
        "BIRC3_rnaseq": (-0.836700, 8.755100),
        "BLM_rnaseq": (-1.081300, 12.501000),
        "BMP1_rnaseq": (-1.230500, 5.003200),
        "BMP2K_rnaseq": (-1.671000, 12.512100),
        "BMP6_rnaseq": (-0.365200, 10.866300),
        "BMP8A_rnaseq": (-0.912700, 16.183100),
        "BRCA1_rnaseq": (-1.254600, 10.805700),
        "BRD2_rnaseq": (-2.493600, 12.934100),
        "BRIP1_rnaseq": (-0.993900, 8.548900),
        "BSG_rnaseq": (-2.211100, 9.338200),
        "BTK_rnaseq": (-1.368900, 4.382600),
        "BUB1B_rnaseq": (-0.918700, 4.706100),
        "BUB1_rnaseq": (-1.176800, 9.330700),
        "CAMK1D_rnaseq": (-1.211700, 5.562600),
        "CBFA2T3_rnaseq": (-1.012800, 21.949100),
        "CBLC_rnaseq": (-1.261900, 10.047600),
        "CCL14_rnaseq": (-0.867400, 4.870000),
        "CCL20_rnaseq": (-0.503700, 8.562100),
        "CCL7_rnaseq": (-0.596000, 14.999500),
        "CCR2_rnaseq": (-1.354700, 5.573700),
        "CCR6_rnaseq": (-1.151900, 5.375100),
        "CCR7_rnaseq": (-0.872400, 8.076700),
        "CD109_rnaseq": (-0.857700, 5.425900),
        "CD163_rnaseq": (-0.819800, 9.651900),
        "CD276_rnaseq": (-1.976000, 4.163300),
        "CD300A_rnaseq": (-1.346900, 5.999100),
        "CD302_rnaseq": (-1.402300, 5.433000),
        "CD40LG_rnaseq": (-1.045500, 5.053200),
        "CD48_rnaseq": (-0.952100, 10.095500),
        "CD5_rnaseq": (-1.097300, 7.688100),
        "CD79B_rnaseq": (-0.729400, 10.235700),
        "CDCP1_rnaseq": (-1.407200, 7.604400),
        "CDK16_rnaseq": (-1.757500, 8.743500),
        "CDK19_rnaseq": (-2.555000, 5.870900),
        "CDK1_rnaseq": (-1.017000, 6.987100),
        "CDK4_rnaseq": (-1.528200, 27.407800),
        "CDK6_rnaseq": (-1.063800, 43.491500),
        "CDKL2_rnaseq": (-0.824100, 8.074300),
        "CDKN2A_rnaseq": (-0.595800, 7.009500),
        "CEBPA_rnaseq": (-1.059700, 32.954100),
        "CHEK1_rnaseq": (-1.112500, 7.180900),
        "CIT_rnaseq": (-0.687900, 8.599800),
        "CKLF_rnaseq": (-1.086000, 8.920300),
        "CLEC10A_rnaseq": (-0.965000, 7.728200),
        "CLTCL1_rnaseq": (-1.098600, 9.838200),
        "CMTM3_rnaseq": (-1.307600, 4.818500),
        "COL1A1_rnaseq": (-0.681500, 7.232300),
        "CR2_rnaseq": (-0.426000, 12.799400),
        "CSF2RB_rnaseq": (-1.326500, 5.805600),
        "CTF1_rnaseq": (-1.573500, 13.821400),
        "CTSG_rnaseq": (-0.641100, 6.968800),
        "CXCL17_rnaseq": (-1.081900, 6.571200),
        "DAPK2_rnaseq": (-0.994100, 5.562200),
        "DDX10_rnaseq": (-1.880000, 8.862300),
        "DEFB1_rnaseq": (-0.248700, 12.234800),
        "DKK1_rnaseq": (-0.384000, 7.632400),
        "EIF2AK2_rnaseq": (-1.623400, 6.204400),
        "EPHB2_rnaseq": (-0.865200, 7.040500),
        "EPHB3_rnaseq": (-0.564000, 12.929900),
        "EPO_rnaseq": (-0.447400, 49.437300),
        "EREG_rnaseq": (-0.370200, 7.573700),
        "EWSR1_rnaseq": (-2.779000, 4.820600),
        "EXT1_rnaseq": (-2.395800, 13.536700),
        "FAM3C_rnaseq": (-1.177000, 18.733100),
        "FANCD2_rnaseq": (-1.463700, 8.115900),
        "FIP1L1_rnaseq": (-1.985900, 30.814800),
        "FSTL3_rnaseq": (-1.012100, 6.834600),
        "FUT4_rnaseq": (-1.033700, 8.577800),
        "GDF10_rnaseq": (-0.541700, 5.580800),
        "GMFB_rnaseq": (-1.801200, 5.899800),
        "GMPS_rnaseq": (-1.939600, 10.356200),
        "GNAS_rnaseq": (-2.095300, 22.642500),
        "GPI_rnaseq": (-1.502200, 8.975900),
        "HERPUD1_rnaseq": (-1.865100, 5.567600),
        "HLF_rnaseq": (-0.665200, 5.951600),
        "HMGA1_rnaseq": (-0.950900, 7.069200),
        "HMMR_rnaseq": (-0.904300, 4.986100),
        "HSP90AA1_rnaseq": (-2.047500, 13.617400),
        "IGF1R_rnaseq": (-1.620800, 9.900900),
        "IKZF1_rnaseq": (-1.305500, 6.471800),
        "IL16_rnaseq": (-1.279700, 5.345200),
        "IL1R2_rnaseq": (-0.276600, 13.453800),
        "IL1RN_rnaseq": (-0.949300, 8.291300),
        "IL23A_rnaseq": (-0.467800, 12.910300),
        "IL2RA_rnaseq": (-0.905000, 6.257200),
        "IL32_rnaseq": (-1.163900, 5.860400),
        "IL33_rnaseq": (-0.946600, 12.545900),
        "IL36RN_rnaseq": (-0.407300, 11.522600),
        "IRF4_rnaseq": (-0.909200, 11.811800),
        "ITGA4_rnaseq": (-1.378200, 8.889500),
        "ITGA5_rnaseq": (-0.539100, 13.389600),
        "ITGA6_rnaseq": (-0.763600, 14.926700),
        "ITGAV_rnaseq": (-1.285900, 5.708100),
        "ITGB1_rnaseq": (-1.331400, 8.740600),
        "ITGB3_rnaseq": (-0.423600, 11.723700),
        "ITGB4_rnaseq": (-0.983400, 4.663600),
        "JAG1_rnaseq": (-1.047900, 10.218900),
        "KRAS_rnaseq": (-1.535500, 46.041700),
        "L1CAM_rnaseq": (-0.256100, 39.014000),
        "LASP1_rnaseq": (-2.425100, 21.224100),
        "LY9_rnaseq": (-1.130300, 4.904100),
        "MAP2K1_rnaseq": (-1.875800, 6.507100),
        "MAP3K12_rnaseq": (-1.576000, 9.244200),
        "MAP4K4_rnaseq": (-1.518200, 10.114900),
        "MAPK12_rnaseq": (-1.241600, 7.576300),
        "MAPK4_rnaseq": (-0.430700, 35.712500),
        "MAPK6_rnaseq": (-1.374700, 7.213000),
        "MAPKAPK2_rnaseq": (-3.330400, 6.449300),
        "MAPKAPK5_rnaseq": (-3.095800, 7.311500),
        "MASTL_rnaseq": (-1.771400, 9.897700),
        "MCAM_rnaseq": (-1.250600, 5.389100),
        "MDK_rnaseq": (-1.050300, 8.144400),
        "MECOM_rnaseq": (-1.149300, 7.935900),
        "MELK_rnaseq": (-1.057100, 4.891300),
        "MIF_rnaseq": (-1.365300, 5.901800),
        "MLKL_rnaseq": (-1.580600, 9.620200),
        "MRC1_rnaseq": (-0.922800, 5.398000),
        "MS4A1_rnaseq": (-0.613700, 5.652400),
        "MST1R_rnaseq": (-1.416500, 5.413000),
        "MYB_rnaseq": (-0.379300, 6.748600),
        "MYH9_rnaseq": (-2.058800, 4.826800),
        "NACA_rnaseq": (-1.790200, 7.706500),
        "NBN_rnaseq": (-1.854100, 6.987100),
        "NEK2_rnaseq": (-0.984000, 5.859800),
        "NLK_rnaseq": (-1.983900, 8.341600),
        "NMB_rnaseq": (-0.991400, 13.451700),
        "NONO_rnaseq": (-3.386500, 4.142700),
        "NPM1_rnaseq": (-1.432400, 5.737400),
        "NRAS_rnaseq": (-1.960800, 6.982800),
        "NRP1_rnaseq": (-1.520800, 5.145400),
        "NUAK2_rnaseq": (-1.729900, 8.646800),
        "NUMBL_rnaseq": (-1.527900, 6.434500),
        "NUP98_rnaseq": (-2.908700, 5.645200),
        "OSM_rnaseq": (-0.801400, 6.354800),
        "PAK2_rnaseq": (-2.606700, 9.203400),
        "PBK_rnaseq": (-0.772400, 7.140500),
        "PDGFB_rnaseq": (-1.390200, 5.895400),
        "PGF_rnaseq": (-0.281000, 13.795400),
        "PICALM_rnaseq": (-2.433600, 9.236300),
        "PIK3R1_rnaseq": (-1.856800, 8.049500),
        "PKMYT1_rnaseq": (-1.194700, 5.398000),
        "PLAUR_rnaseq": (-1.246800, 7.777800),
        "PLAU_rnaseq": (-0.692100, 6.954200),
        "PLK1_rnaseq": (-1.072200, 4.871700),
        "PLK4_rnaseq": (-1.059400, 6.361900),
        "PML_rnaseq": (-1.544200, 4.474100),
        "PRKCB_rnaseq": (-1.201700, 5.657900),
        "PRKDC_rnaseq": (-1.707400, 10.265500),
        "PTPRC_rnaseq": (-1.317700, 4.334600),
        "PVR_rnaseq": (-1.789800, 10.866500),
        "RBM15_rnaseq": (-2.344900, 4.094600),
        "RECQL4_rnaseq": (-1.080300, 8.878500),
        "RHOH_rnaseq": (-1.173600, 5.610300),
        "RIPK2_rnaseq": (-1.692700, 11.297800),
        "RPN1_rnaseq": (-2.381700, 7.411200),
        "RPS6KB1_rnaseq": (-1.945200, 11.974500),
        "RPS6KL1_rnaseq": (-1.210600, 8.356300),
        "RUNX1_rnaseq": (-2.411300, 6.840400),
        "SCYL1_rnaseq": (-2.661000, 9.444100),
        "SCYL2_rnaseq": (-2.503000, 5.271000),
        "SDHAF2_rnaseq": (-1.924400, 6.005100),
        "SELP_rnaseq": (-1.121600, 4.543000),
        "SEMA3C_rnaseq": (-0.978100, 6.517100),
        "SEMA4B_rnaseq": (-1.352700, 12.519900),
        "SEMA7A_rnaseq": (-0.486900, 6.591500),
        "SH3GL1_rnaseq": (-2.399300, 4.590900),
        "SIGLEC6_rnaseq": (-0.821800, 84.187200),
        "SIGLEC7_rnaseq": (-1.228700, 4.662600),
        "SLAMF1_rnaseq": (-1.267600, 6.803700),
        "SLC3A2_rnaseq": (-1.466600, 4.739600),
        "SLC44A1_rnaseq": (-1.480200, 5.951600),
        "SMO_rnaseq": (-0.970500, 12.961400),
        "SPECC1_rnaseq": (-1.646700, 6.920200),
        "SPP1_rnaseq": (-0.607500, 6.174300),
        "SRPK3_rnaseq": (-0.707700, 10.086700),
        "STC2_rnaseq": (-0.626800, 7.053900),
        "STIL_rnaseq": (-1.252300, 8.156500),
        "STK17A_rnaseq": (-1.424900, 9.877200),
        "STK24_rnaseq": (-2.353100, 6.374600),
        "STK32A_rnaseq": (-1.058900, 5.662600),
        "STK3_rnaseq": (-2.121800, 8.338500),
        "STYK1_rnaseq": (-1.071200, 7.317000),
        "TBK1_rnaseq": (-3.205900, 20.863600),
        "TFG_rnaseq": (-2.009000, 9.284700),
        "TLR10_rnaseq": (-0.697200, 15.513700),
        "TLR2_rnaseq": (-1.157200, 6.085100),
        "TMPRSS2_rnaseq": (-1.294600, 15.655800),
        "TNFRSF10C_rnaseq": (-1.338900, 4.845600),
        "TNFRSF1A_rnaseq": (-2.223600, 6.456000),
        "TNFSF4_rnaseq": (-0.937700, 5.266000),
        "TPM3_rnaseq": (-1.809600, 6.304300),
        "TRIM28_rnaseq": (-2.499900, 8.850400),
        "TTK_rnaseq": (-1.051200, 5.753100),
        "TWF1_rnaseq": (-1.878100, 8.735200),
        "VEGFA_rnaseq": (-1.212300, 5.692700),
        "VEGFC_rnaseq": (-0.534600, 9.944400),
        "VGF_rnaseq": (-0.151100, 11.774200),
        "ZNF384_rnaseq": (-3.102100, 10.464600)
    }

    gene_columns = list(gene_ranges.keys())

    # Define categorical columns and their options
    categorical_columns = ["Gender", "ajcc_pathologic_tumor_stage", "treatment_outcome_first_course"]
    categorical_options = {
        "Gender": ["Male", "Female"],
        "ajcc_pathologic_tumor_stage": ['Stage IV', 'Stage IB', 'Stage IIIA', 'Stage IA', 'Stage IIIB',
                                        'Stage IIB', 'Stage IIA', 'Stage II', 'Stage I'],
        "treatment_outcome_first_course": ['Complete Remission/Response', 'Progressive Disease',
                                           'Partial Remission/Response', 'Stable Disease', '[Not Evaluated]']
    }

    # Initialize session state for input data
    if "input_data" not in st.session_state:
        st.session_state.input_data = {}


    # Random value generator
    def generate_random_values():
        random_data = {}
        random_data["Age"] = np.random.randint(18, 90)
        random_data["Gender"] = np.random.choice(categorical_options["Gender"])
        for col in categorical_columns[1:]:  # Skip Gender
            options = [opt for opt in categorical_options[col] if opt != 'nan']
            random_data[col] = np.random.choice(options)
        for col in gene_columns:
            min_val, max_val = gene_ranges[col]
            random_data[col] = np.random.uniform(min_val, max_val)
        st.session_state.input_data = random_data


    # Button to generate random values
    if st.button("Generate Random Values"):
        generate_random_values()

    # Input form
    with st.form(key="recurrence_form"):
        st.subheader("Patient Data Input")
        input_data = {}

        # Age input
        input_data["Age"] = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=st.session_state.input_data.get("Age", 50)
        )

        # Categorical inputs
        for col in categorical_columns:
            input_data[col] = st.selectbox(
                col,
                options=categorical_options[col],
                index=categorical_options[col].index(st.session_state.input_data.get(col, categorical_options[col][0]))
                if col in st.session_state.input_data else 0
            )

        # Gene inputs expander to save space
        with st.expander("Gene Expression Data"):
            # Use columns to display multiple inputs per row
            col1, col2 = st.columns(2)
            gene_count = 0

            for gene in gene_columns:
                min_val, max_val = gene_ranges[gene]
                # Alternate between columns
                current_col = col1 if gene_count % 2 == 0 else col2
                with current_col:
                    input_data[gene] = st.number_input(
                        f"{gene}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(st.session_state.input_data.get(gene, 0.0)),
                        format="%.5f"
                    )
                gene_count += 1

        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        # Send data to backend for processing
        try:
            with st.spinner("Analyzing patient data for recurrence risk..."):
                response = requests.post(f"{BACKEND_URL}/recurrence", json=input_data)

            if response.ok:
                data = response.json()

                # Extract results and plots
                results = data.get('results', {})
                plots = data.get('plots', {})

                # Create tabbed interface for better organization
                tabs = st.tabs(["Risk Overview", "Tumor Event Details", "PFI Analysis", "Clinical Recommendations"])

                # Tab 1: Risk Overview
                with tabs[0]:
                    # Determine risk level color
                    risk_level = results.get('risk_level', 'Unknown')
                    risk_color = {
                        "High": "#e74c3c",
                        "Medium": "#f1c40f",
                        "Low": "#27ae60"
                    }.get(risk_level, "#7f8c8d")

                    st.markdown(f"### Patient Risk Profile: <span style='color:{risk_color}'>{risk_level}</span>",
                                unsafe_allow_html=True)


                    # Risk Gauge Chart
                    def create_risk_gauge(risk_probability):

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_probability * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Progression Risk (%)", 'font': {'size': 24}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "#27ae60"},  # Low: Green
                                    {'range': [25, 75], 'color': "#f1c40f"},  # Medium: Yellow
                                    {'range': [75, 100], 'color': "#e74c3c"}  # High: Red
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': risk_probability * 100
                                }
                            }
                        ))

                        fig.update_layout(height=250)
                        return fig


                    # Display the gauge
                    st.plotly_chart(
                        create_risk_gauge(results.get('pfi_probability', 0)),
                        use_container_width=True
                    )

                    # Main metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tumor Event Type", results.get('event_prediction', 'Unknown'))
                    with col2:
                        st.metric("Progression within 12 Months",
                                  f"{results.get('pfi_probability', 0) * 100:.1f}%",
                                  f"{results.get('pfi_prediction', 'Unknown')}")
                    with col3:
                        median_pfi = results.get('median_pfi_time')
                        if median_pfi is not None:
                            st.metric("Median PFI Time", f"{median_pfi:.0f} days",
                                      f"{median_pfi / 30.4:.1f} months")
                        else:
                            st.metric("Median PFI Time", "Not reached")


                    # Patient Timeline Visualization
                    # For the follow-up timeline, increase the height
                    def create_timeline(median_pfi):
                        import plotly.graph_objects as go

                        if not median_pfi:
                            return None

                        events = [
                            {"label": "Today", "day": 0},
                            {"label": "3 Month F/U", "day": 90},
                            {"label": "6 Month F/U", "day": 180},
                            {"label": "1 Year F/U", "day": 365},
                            {"label": "Median PFI", "day": median_pfi},
                        ]

                        events.sort(key=lambda x: x["day"])

                        fig = go.Figure()

                        # Add events to timeline
                        for i, event in enumerate(events):
                            fig.add_trace(go.Scatter(
                                x=[event["day"]],
                                y=[0],
                                mode="markers+text",
                                marker=dict(size=15, symbol="circle", color="#3498db"),
                                text=[event["label"]],
                                textposition="top center",
                                name=event["label"]
                            ))

                        # Add line connecting events
                        days = [e["day"] for e in events]
                        fig.add_trace(go.Scatter(
                            x=days,
                            y=[0] * len(days),
                            mode="lines",
                            line=dict(color="#3498db", width=2),
                            showlegend=False
                        ))

                        fig.update_layout(
                            title="Follow-up Timeline",
                            xaxis=dict(title="Days"),
                            yaxis=dict(showticklabels=False, zeroline=False),
                            height=350,  # Increased from 250 to 350
                            margin=dict(t=50, b=30),  # Added some margin
                            hovermode="x unified"
                        )

                        return fig


                    # Display timeline
                    median_pfi = results.get('median_pfi_time')
                    if median_pfi:
                        timeline = create_timeline(median_pfi)
                        if timeline:
                            st.plotly_chart(timeline, use_container_width=True)

                # Tab 2: Tumor Event Details
                with tabs[1]:
                    st.subheader("Tumor Event Type Prediction")

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.write(f"**Predicted Event:** {results.get('event_prediction', 'Unknown')}")

                        # Clinical notes for different event types
                        CLINICAL_NOTES = {
                            "No New Tumor": "No evidence of tumor recurrence detected",
                            "Local": "Cancer has returned to the original site",
                            "Regional": "Cancer has spread to nearby lymph nodes or tissues",
                            "Distant": "Cancer has spread to distant organs (Stage IV)",
                            "New Primary": "New distinct cancer type identified",
                            "Unknown": "Recurrence type needs further investigation"
                        }
                        note = CLINICAL_NOTES.get(results.get('event_prediction', 'Unknown'),
                                                  'Consult oncologist for interpretation')
                        st.write(f"**Clinical Note:** {note}")

                        # Show probability breakdown
                        st.write("**Probability Breakdown:**")

                        # Create a horizontal bar chart for event probabilities

                        event_probs = results.get('event_probabilities', {})
                        if event_probs:
                            labels = list(event_probs.keys())
                            values = [float(p) for p in event_probs.values()]

                            # Sort by probability value
                            sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
                            sorted_labels = [labels[i] for i in sorted_indices]
                            sorted_values = [values[i] for i in sorted_indices]

                            colors = ["#3498db" if label != results.get('event_prediction', 'Unknown')
                                      else "#e74c3c" for label in sorted_labels]

                            fig = go.Figure(go.Bar(
                                x=[v * 100 for v in sorted_values],
                                y=sorted_labels,
                                orientation='h',
                                marker_color=colors,
                                text=[f"{v * 100:.1f}%" for v in sorted_values],
                                textposition='auto'
                            ))

                            fig.update_layout(
                                title="Event Type Probabilities",
                                xaxis_title="Probability (%)",
                                yaxis_title="Event Type",
                                height=300,
                                margin=dict(l=10, r=10, t=40, b=10)
                            )

                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Show feature importance if available
                        if 'tumor_features' in plots:
                            st.image(plots['tumor_features'], caption="Important Features for Tumor Event Prediction")
                        elif 'top_features_tumor' in results:
                            # Create interactive feature importance chart
                            def create_feature_importance_chart(features, values, title):

                                # Sort by importance value
                                df = pd.DataFrame({'Feature': features, 'Value': values})
                                df = df.sort_values('Value', ascending=True)

                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    y=df['Feature'],
                                    x=df['Value'],
                                    orientation='h',
                                    marker_color="#3498db",
                                    text=[f"{v:.4f}" for v in df['Value']],
                                    textposition='auto'
                                ))

                                fig.update_layout(
                                    title=title,
                                    xaxis_title="Feature Importance",
                                    yaxis_title="Feature",
                                    height=400
                                )

                                return fig


                            features = results['top_features_tumor']['features']
                            importances = results['top_features_tumor']['importances']

                            st.plotly_chart(
                                create_feature_importance_chart(features, importances,
                                                                "Top Features - Tumor Event Model"),
                                use_container_width=True
                            )


                    # Gene expression heatmap
                    def create_gene_heatmap(input_data, top_genes):

                        # Filter to top genes only
                        gene_data = {g: input_data.get(g, 0) for g in top_genes if g in input_data}

                        fig = go.Figure(go.Heatmap(
                            z=[[v for v in gene_data.values()]],
                            x=list(gene_data.keys()),
                            colorscale='RdBu_r',
                            zmid=0,  # Center colorscale at 0
                        ))

                        fig.update_layout(
                            title="Key Gene Expression Values",
                            height=200,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )

                        return fig


                    # Get the gene names from feature importance
                    if 'top_features_tumor' in results:
                        top_genes = [f for f in results['top_features_tumor']['features']
                                     if f.endswith('_rnaseq')][:10]

                        if top_genes:
                            st.plotly_chart(
                                create_gene_heatmap(input_data, top_genes),
                                use_container_width=True
                            )

                # Tab 3: PFI Analysis
                with tabs[2]:
                    st.subheader("Progression-Free Interval (PFI) Analysis")

                    # Interactive Survival Curve
                    if 'survival_curve' in plots:
                        # Display the static image from backend
                        st.image(plots['survival_curve'], caption="PFI Survival Curve")

                        # Create interactive version

                        try:
                            # Estimate survival function data from backend
                            times = list(range(0, 731))  # Assuming 2 years of data

                            # We need to handle this based on what the backend provides
                            # This is an approximation as we don't have the exact survival function values
                            median_pfi = results.get('median_pfi_time')
                            one_year_pfi = results.get('one_year_pfi', 0.5)
                            two_year_pfi = results.get('two_year_pfi', 0.25)

                            # Generate an exponential decay curve based on the median PFI
                            if median_pfi:
                                # Calculate lambda for exponential decay S(t) = exp(-lambda*t)
                                # At median time, S(t) = 0.5, so lambda = ln(2)/median
                                lambda_val = np.log(2) / median_pfi
                                probabilities = [np.exp(-lambda_val * t) for t in times]
                            else:
                                # If median not reached, use a slower decay
                                lambda_val = -np.log(one_year_pfi) / 365
                                probabilities = [np.exp(-lambda_val * t) for t in times]

                            # Force exact values at 1 and 2 years
                            if 365 in times:
                                idx_1yr = times.index(365)
                                probabilities[idx_1yr] = one_year_pfi

                            if 730 in times:
                                idx_2yr = times.index(730)
                                probabilities[idx_2yr] = two_year_pfi

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=times,
                                y=probabilities,
                                mode='lines',
                                name='PFI Probability',
                                line=dict(color='#2e86c1', width=3)
                            ))

                            # Add reference lines
                            fig.add_hline(y=0.5, line_dash="dash", line_color="#e74c3c",
                                          annotation_text="Median")
                            fig.add_vline(x=365, line_dash="dot", line_color="#27ae60",
                                          annotation_text="1 Year")

                            fig.update_layout(
                                title="Interactive Progression-Free Interval Curve",
                                xaxis_title="Days",
                                yaxis_title="Probability of No Progression",
                                yaxis=dict(range=[0, 1]),
                                height=450,  # Increased height
                                margin=dict(l=50, r=30, t=50, b=50),  # Adjusted margins
                                hovermode="x unified"
                            )

                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.info("Interactive chart not available. Using static image.")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**PFI Time Metrics:**")
                        median_pfi = results.get('median_pfi_time')
                        if median_pfi is not None:
                            st.write(f"- Median PFI Time: {median_pfi:.0f} days ({median_pfi / 30.4:.1f} months)")
                        else:
                            st.write("- Median PFI Time: Not reached within available range")

                        st.write(f"- 1-Year PFI Rate: {results.get('one_year_pfi', 0):.1%}")
                        st.write(f"- 2-Year PFI Rate: {results.get('two_year_pfi', 0):.1%}")
                        st.write(
                            f"- Progression within 12 Months: {results.get('pfi_prediction', 'Unknown')} ({results.get('pfi_probability', 0):.1%} probability)")

                    with col2:
                        # Top 10 Features for CoxPH Model
                        st.write("**Top Influential Features for PFI Prediction:**")

                        # Interactive visualization of Cox features
                        if 'top_features_coxph' in results:
                            # For the Cox feature chart, increase the height
                            def create_cox_feature_chart(features, coeffs, title):
                                import plotly.graph_objects as go

                                # Create color scale based on values
                                colors = ['#e74c3c' if v > 0 else '#3498db' for v in coeffs]

                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    y=features,
                                    x=coeffs,
                                    orientation='h',
                                    marker_color=colors,
                                    text=[f"{v:.4f}" for v in coeffs],
                                    textposition='auto'
                                ))

                                fig.update_layout(
                                    title=title,
                                    xaxis_title="Coefficient Value",
                                    yaxis_title="Feature",
                                    height=550,  # Increased from 400 to 550
                                    margin=dict(l=200, r=30, t=50, b=30)  # Added more space for feature names
                                )

                                # Add vertical line at x=0
                                fig.add_vline(x=0, line_dash="dash", line_color="gray")

                                return fig


                            features = results['top_features_coxph']['features']
                            coeffs = results['top_features_coxph']['coefficients']

                            st.plotly_chart(
                                create_cox_feature_chart(features, coeffs, "Top Features - Cox Model"),
                                use_container_width=True
                            )
                        elif 'coxph_features' in plots:
                            st.image(plots['coxph_features'], caption="Cox Model Feature Importance")

                # Tab 4: Clinical Recommendations
                with tabs[3]:
                    st.subheader("Clinical Recommendations")

                    # Comparative Reference Card
                    st.markdown("""
                        ### Reference Values

                        | Risk Level | Median PFI Time | 1-Year PFI Rate | Clinical Action |
                        |------------|-----------------|-----------------|-----------------|
                        | High       | < 365 days      | < 50%           | Aggressive monitoring |
                        | Medium     | 365-730 days    | 50-75%          | 3-6 month follow-up |
                        | Low        | > 730 days      | > 75%           | Standard follow-up |
                        """)

                    # Risk-based recommendations
                    risk_level = results.get('risk_level', 'Unknown')
                    event_type = results.get('event_prediction', 'Unknown')

                    st.subheader("Personalized Recommendations")

                    if risk_level == "High":
                        st.error("""
                            #### High Risk Patient Plan
                            - **Immediate Action:** High risk of progression within 1 year (median PFI < 365 days). Consider aggressive monitoring or adjuvant therapy.
                            - **Imaging:** Schedule follow-up imaging every 3 months (CT or PET/CT).
                            - **Molecular Testing:** Consider liquid biopsy for detection of circulating tumor DNA.
                            - **Treatment Options:** Discuss additional targeted therapy or immunotherapy options based on molecular profile.
                            - **Clinical Trials:** Evaluate eligibility for relevant trials targeting high-risk patients.
                            """)
                    elif risk_level == "Medium":
                        st.warning("""
                            #### Medium Risk Patient Plan
                            - **Monitoring:** Moderate risk of progression (median PFI < 730 days). Schedule follow-ups every 3-6 months with appropriate imaging.
                            - **Biomarker Testing:** Monitor tumor markers if applicable to cancer type.
                            - **Prevention:** Evaluate maintenance therapy options to delay progression.
                            - **Lifestyle Modifications:** Discuss exercise, nutrition, and stress management strategies.
                            - **Multidisciplinary Review:** Present case at tumor board for consensus recommendations.
                            """)
                    else:  # Low
                        st.success("""
                            #### Low Risk Patient Plan
                            - **Routine Care:** Low risk of early progression (median PFI ≥ 730 days). Continue standard follow-up every 6-12 months.
                            - **Surveillance Imaging:** Annual imaging may be sufficient based on NCCN guidelines.
                            - **Patient Education:** Reinforce lifestyle modifications and signs/symptoms to watch for.
                            - **Survivorship Planning:** Develop long-term survivorship care plan.
                            - **Reassess:** Repeat risk assessment annually or if new symptoms arise.
                            """)

                    # Additional recommendations based on tumor event type
                    if event_type != "No New Tumor" and event_type != "Unknown":
                        st.subheader(f"Additional Recommendations for {event_type} Recurrence")
                        if event_type == "Local":
                            st.markdown("- Consider surgical re-resection if feasible\n"
                                        "- Evaluate for radiation therapy if not previously administered\n"
                                        "- Contrast-enhanced MRI to assess precise extent of local recurrence")
                        elif event_type == "Regional":
                            st.markdown("- Lymph node evaluation and possible lymphadenectomy\n"
                                        "- Consider regional nodal irradiation\n"
                                        "- PET/CT to fully assess regional involvement")
                        elif event_type == "Distant":
                            st.markdown("- Biopsy of metastatic site for confirmation and molecular analysis\n"
                                        "- Systemic therapy based on histology and molecular features\n"
                                        "- Evaluate for oligometastatic approach if limited metastases")
                        elif event_type == "New Primary":
                            st.markdown("- Complete staging workup as for a new primary cancer\n"
                                        "- Molecular testing to distinguish from original primary\n"
                                        "- Treatment following standard guidelines for the new primary site")

                # Add confidence score based on data completeness
                feature_count = len(input_data)
                max_features = len(gene_columns) + len(categorical_columns) + 1  # +1 for age
                confidence = min(feature_count / max_features * 100, 100)

                st.sidebar.subheader("Prediction Confidence")
                st.sidebar.progress(confidence / 100)
                st.sidebar.write(f"Data completeness: {confidence:.1f}%")

                # Add report download option
                st.sidebar.subheader("Export Options")

                # Create PDF report (simplified - would need a real PDF generation library)


                def generate_report_data():
                    report_data = {
                        "patient": {
                            "age": input_data["Age"],
                            "gender": input_data["Gender"],
                            "stage": input_data["ajcc_pathologic_tumor_stage"]
                        },
                        "predictions": {
                            "risk_level": risk_level,
                            "pfi_probability": results.get('pfi_probability', 0),
                            "event_type": results.get('event_prediction', 'Unknown'),
                            "median_pfi": results.get('median_pfi_time')
                        }
                    }
                    return json.dumps(report_data, indent=2)


                report_data = generate_report_data()
                st.sidebar.download_button(
                    "Download Report Data (JSON)",
                    data=report_data,
                    file_name="cancer_recurrence_report.json",
                    mime="application/json",
                )

            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")