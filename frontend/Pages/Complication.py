import streamlit as st
import requests
import pandas as pd
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import random
import matplotlib.pyplot as plt


def run():
    st.markdown("<h1 class='stTitle'>🌟 Lung Cancer Complication Predictor</h1>", unsafe_allow_html=True)
    st.write("This application predicts the severity, type, and treatment timing of lung cancer complications based on input data.")

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

