import streamlit as st
import pandas as pd
import numpy as np
import joblib



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


# Load the models
model_ctype_catl = joblib.load("random_forest_model_ctype_catl_final2.pkl")
model_ctypel = joblib.load("random_forest_model_ctypel_final3.pkl")
model_comp_gap_category = joblib.load("random_forest_model_treatment_category_final2.pkl")

# Mappings for predictions
ctypel_mapping = {
    'Acute / Chronic Respiratory Failure': 0, 'Atelectasis': 1, 'Bleeding & Wound Healing Issues': 2,
    'Bronchopulmonary Fistula': 3, 'Bronchospasm': 4, 'Cardiac Arrhythmia': 5, 'Cardiac Emergencies': 6,
    'Cerebral vascular accident (CVA) / Stroke': 7, 'Congestive Heart Failure (CHF)': 8,
    'Deep Venous Thrombosis (DVT)': 9, 'Fever Requiring Antibiotics': 10, 'Hospitalization': 11,
    'Hypokalemia': 12, 'Hypotension / Vasovagal Reaction': 13, 'Infectious': 14, 'Other Specify': 15,
    'Pain Requiring Referral to an Anesthesiologist / Pain Specialist': 16, 'Pneumothorax': 17,
    'Pulmonary Embolus / Emboli': 18, 'Respiratory Arrest': 19, 'Rib Fracture(s)': 20, 'Urinary': 21,
    'Vocal Cord Immobility / Paralysis': 22
}

ctype_catl_mapping = {0: "Intermediate", 1: "Major", 2: "Minor"}
comp_gap_category_mapping = {2: "pre treatment", 0: "during treatment", 1: "post treatment"}

# Define all encoding mappings
mappings = {
    'sex': {'Female': 0, 'Male': 1},
    'bmi_curc': {'0-18.5': 0, '18.5-25': 1, '25-30': 2, '30+': 3},
    'cig_stat': {'Current Cigarette Smoker': 0, 'Former Cigarette Smoker': 1, 'Never Smoked Cigarettes': 2},
    'ph_any_trial': {'No': 0, 'Yes': 1},
    'diabetes_f': {'No': 0, 'Yes': 1},
    'hyperten_f': {'No': 0, 'Yes': 1},
    'emphys_f': {'No': 0, 'Yes': 1},
    'bronchit_f': {'No': 0, 'Yes': 1},
    'hearta_f': {'No': 0, 'Yes': 1},
    'proc_numl': {
        'Biopsy & Cytology': 0, 'Biopsy, endobronchial': 1, 'Biopsy, transbronchial': 2,
        'Bone Radiograph': 3, 'Bronchoscopy': 4, 'Bx, other - non-lung histology': 5,
        'CT - abdomen': 6, 'CT - chest, abdomen and pelvis': 7, 'CT Scan - Brain': 8,
        'CT Scan - Chest': 9, 'CT Scan - abdomen and pelvis': 10, 'CT Scan - chest and upper abdomen': 11,
        'CT, MRI & Ultrasound': 12, 'CT-scan, spiral - chest': 13, 'Chest Radiogram - Lat': 14,
        'Clinical Exam': 15, 'Comparison of Chest X-rays': 16, 'Cytology': 17,
        'Internal Referrals': 18, 'Lymphadenectomy': 19, 'MRI Scan - Brain': 20,
        'Mediastinoscopy': 21, 'Other - PET': 22, 'Other - radionucleotide, Fusion PET/CT': 23,
        'Pulmonary Function Tests': 24, 'Radiographic & Miscellaneous': 25,
        'Radionuclide Scan - Bone': 26, 'Record review': 27, 'Resection': 28,
        'Surgical Open Biopsy': 29, 'Thoracentesis': 30, 'Thoracoscopy': 31,
        'Thoracotomy': 32, 'Transbronchial Aspiration': 33, 'Transthoracic Aspiration': 34,
        'Ventilation perfusion lung scan': 35
    },
    'del_invas_cat': {
        'Bronchoscopy with biopsy': 0, 'Bronchoscopy without biopsy': 1, 'Chest Imaging': 2,
        'Chest X-ray': 3, 'Clinical': 4, 'Comparison': 5, 'Cytology': 6, 'Mediastinoscopy': 7,
        'Needle biopsy': 8, 'Other with biopsy': 9, 'Other- no biopsy': 10, 'Other- non-lung': 11,
        'PET Scan': 12, 'Resection - no approach specified': 13, 'Staging Imaging': 14,
        'Thoracentesis': 15, 'Thoracoscopy': 16, 'Thoracotomy': 17
    },
    'biop': {'No': 0, 'Yes': 1},
    'biopllink0': {'No': 0, 'Yes': 1},
    'reasfolll': {'No': 0, 'Yes': 1},
    'lung_stage': {
        'Stage IA': 0, 'Stage IB': 1, 'Stage IIA': 2, 'Stage IIB': 3,
        'Stage IIIA': 4, 'Stage IIIB': 5, 'Stage IV': 6
    },
    'lung_clinstage': {
        'Occult Carcinoma': 0, 'Stage IA': 1, 'Stage IB': 2, 'Stage IIA': 3,
        'Stage IIB': 4, 'Stage IIIA': 5, 'Stage IIIB': 6, 'Stage IV': 7
    },
    'lung_stage_t': {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3},
    'lung_stage_n': {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'NX': 4},
    'lung_stage_m': {'M0': 0, 'M1': 1},
    'lung_histtype_cat': {
        'Adenocarcinoma': 0, 'Bronchiolo-alveolar carcinoma': 1, 'Carcinoma, NOS': 2,
        'Large cell carcinoma': 3, 'Other NSC carcinoma': 4, 'Other/Missing': 5,
        'Squamous cell carcinoma': 6
    },
    'trt_familyl': {
        'Chemotherapy': 0, 'Non-curative treatment': 1, 'Pneumonectomy or bilobectomy': 2,
        'Radiation treatment': 3, 'Wedge resection, segmental resection, or lobectomy': 4
    },
    'trt_numl': {
        'Bilobectomy': 0, 'Chemotherapy - Platinum-Based Drugs': 1, 'Chest wall resection': 2,
        'External photon beam': 3, 'Lobectomy': 4, 'Lymphadenectomy / lymph node sampling': 5,
        'Other chemotherapy (specify)': 6, 'Other treatment, NOS': 7, 'Partial pleurectomy': 8,
        'Pneumonectomy': 9, 'Radiation Therapy (General & Unspecified)': 10,
        'Segmental resection': 11, 'Surgical Procedures': 12, 'Systemic treatment, NOS': 13,
        'Thoracentesis': 14, 'Wedge resection': 15
    },
    'neoadjuvant': {'Neoadjuvant': 0, 'Not neoadjuvant': 1}
}

def map_input(field_name, value):
    """Map input value to encoded number using predefined mappings"""
    return mappings[field_name][value]

def user_input_form():

    st.header("Patient Data Entry")
    inputs = {}
    
    # Create columns for better layout
    col1, col2= st.columns(2)
    
    with col1:
        inputs['age'] = st.number_input("Age")
        inputs['sex'] = map_input('sex', st.selectbox("Sex", list(mappings['sex'].keys())))
        inputs['bmi_curc'] = map_input('bmi_curc', st.selectbox("BMI Category", list(mappings['bmi_curc'].keys())))
        inputs['cig_stat'] = map_input('cig_stat', st.selectbox("Smoking Status", list(mappings['cig_stat'].keys())))
        inputs['pack_years'] = st.number_input("Pack Years", min_value=0.0, value=0.0)
        inputs['ph_any_trial'] = map_input('ph_any_trial', st.selectbox("Clinical Trial Participation", ['No', 'Yes']))
        inputs['diabetes_f'] = map_input('diabetes_f', st.selectbox("Diabetes", ['No', 'Yes']))
        inputs['hyperten_f'] = map_input('hyperten_f', st.selectbox("Hypertension", ['No', 'Yes']))

    with col2:
        inputs['emphys_f'] = map_input('emphys_f', st.selectbox("Emphysema", ['No', 'Yes']))
        inputs['bronchit_f'] = map_input('bronchit_f', st.selectbox("Chronic Bronchitis", ['No', 'Yes']))
        inputs['hearta_f'] = map_input('hearta_f', st.selectbox("Heart Disease", ['No', 'Yes']))
        inputs['proc_numl'] = map_input('proc_numl', st.selectbox("Procedure Type", list(mappings['proc_numl'].keys())))
        inputs['del_invas_cat'] = map_input('del_invas_cat', st.selectbox("Diagnostic Method", list(mappings['del_invas_cat'].keys())))
        inputs['biop'] = map_input('biop', st.selectbox("Biopsy Performed", ['No', 'Yes']))
        inputs['biopllink0'] = map_input('biopllink0', st.selectbox("Biopsy Linked", ['No', 'Yes']))
        inputs['reasfolll'] = map_input('reasfolll', st.selectbox("Follow-up Required", ['No', 'Yes']))

    # Second row of columns
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Cancer Staging")
        inputs['lung_stage'] = map_input('lung_stage', st.selectbox("Overall Stage", list(mappings['lung_stage'].keys())))
        inputs['lung_clinstage'] = map_input('lung_clinstage', st.selectbox("Clinical Stage", list(mappings['lung_clinstage'].keys())))
        inputs['lung_stage_t'] = map_input('lung_stage_t', st.selectbox("T Stage", list(mappings['lung_stage_t'].keys())))
        inputs['lung_stage_n'] = map_input('lung_stage_n', st.selectbox("N Stage", list(mappings['lung_stage_n'].keys())))
        inputs['lung_stage_m'] = map_input('lung_stage_m', st.selectbox("M Stage", list(mappings['lung_stage_m'].keys())))
        
    with col4:
        st.subheader("Treatment Details")
        inputs['lung_histtype_cat'] = map_input('lung_histtype_cat', st.selectbox("Histology Type", list(mappings['lung_histtype_cat'].keys())))
        inputs['trt_familyl'] = map_input('trt_familyl', st.selectbox("Treatment Category", list(mappings['trt_familyl'].keys())))
        inputs['trt_numl'] = map_input('trt_numl', st.selectbox("Specific Treatment", list(mappings['trt_numl'].keys())))
        inputs['neoadjuvant'] = map_input('neoadjuvant', st.selectbox("Neoadjuvant Therapy", ['Neoadjuvant', 'Not neoadjuvant']))

    return pd.DataFrame([inputs])


    return pd.DataFrame([inputs])

# Main app
st.title("Lung Cancer Complication Predictor")
st.write("Predicts complication severity, type, and treatment timing")

# Get inputs
input_df = user_input_form()


# Create centered columns for the button
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio for different screen sizes

# Prediction logic
if st.button("Predict"):
    try:
        # Convert to float for model compatibility
        input_processed = input_df.astype(float)
        
        # Make predictions
        pred_catl = model_ctype_catl.predict(input_processed)
        pred_ctypel = model_ctypel.predict(input_processed)
        pred_gap = model_comp_gap_category.predict(input_processed)
        
        # Get probabilities for ctypel
        pred_ctypel_proba = model_ctypel.predict_proba(input_processed)[0]
        top_5_indices = np.argsort(pred_ctypel_proba)[-5:][::-1]
        top_5 = [(list(ctypel_mapping.keys())[idx], pred_ctypel_proba[idx]) for idx in top_5_indices]

        # Create a centered container
        with st.container():
            # Center the content using columns
            col1, col2, col3 = st.columns([1, 3, 1])  # Adjust the ratio for different screen sizes
            
            with col2:
                # Add visual separation
                st.markdown("---")
                
                # Display results in cards
                with st.expander("### Prediction Results", expanded=True):
                    st.markdown(f"""
                    **Severity:**  
                    <span style="color: #2e86c1; font-size: 20px">{ctype_catl_mapping[pred_catl[0]]}</span>  
                    
                    **Complication Type:**  
                    <span style="color: #2e86c1; font-size: 20px">{list(ctypel_mapping.keys())[pred_ctypel[0]]}</span>  
                    
                    **Treatment Timing:**  
                    <span style="color: #2e86c1; font-size: 20px">{comp_gap_category_mapping[pred_gap[0]]}</span>
                    """, unsafe_allow_html=True)
                
                # Add space between sections
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Top 5 probabilities in a styled box
                with st.container():
                    st.markdown("### Top 5 Complication Probabilities")
                    for name, prob in top_5:
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; margin: 5px 0; 
                                    background-color: black; border-left: 4px solid #2e86c1">
                            <strong>{name}:</strong> {prob*100:.2f}%
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add final spacing
                st.markdown("---")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")