from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
import zipfile
import tempfile
import torch

from TNM.models.Files.dicom_processing import load_dicom_images, convert_to_3d_volume, normalize_hu, fast_resize
from TNM.models.Files.model_utils import load_models

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Cox model and optional patient data
# Adjust paths as needed
cox_model = joblib.load('Prognosis/model/cox_model.pkl')
# Load models once when starting app
model_ctype_catl = joblib.load("models/tabnet_cat_ctypl.pkl")
model_ctypel = joblib.load("models/tabnet_ctypel.pkl")
model_comp_gap_category = joblib.load("models/tabnet_trt_cat.pkl")

# Load models once at startup
model_t, model_n, model_m, model_loc = load_models()
model_t.to(device).eval()
model_n.to(device).eval()
model_m.to(device).eval()
model_loc.to(device).eval()

# Mapping dictionaries (as per your Streamlit code)
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
inv_ctypel_mapping = {v: k for k, v in ctypel_mapping.items()}

ctype_catl_mapping = {0: "Intermediate", 1: "Major", 2: "Minor"}
comp_gap_category_mapping = {2: "pre treatment", 0: "during treatment", 1: "post treatment"}

# Input feature mappings (for encoding categorical fields)
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
    return mappings[field_name][value]















try:
    patients_data = pd.read_csv('Prognosis/df_train_scaled_external_tested.csv', index_col=0)
except FileNotFoundError:
    patients_data = None

# Scaler for risk score normalization
scaler = MinMaxScaler(feature_range=(0, 1))


def fig_to_base64(fig):
    """
    Convert a Matplotlib figure to a base64-encoded PNG string.
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.getvalue()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


@app.route('/prognosis', methods=['POST'])
def prognosis():
    # Expecting a file upload with key 'file'
    if 'file' not in request.files:
        return 'No file part in request', 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception:
        return 'Invalid CSV file', 400

    # Validate single row
    if df.shape[0] != 1:
        return 'Please upload exactly one row of data.', 400

    # Ensure all expected features are present
    expected_cols = cox_model.params_.index.tolist()
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        return f"Missing columns: {', '.join(missing)}", 400

    # Align columns
    df = df[expected_cols]

    # Predict survival function and plot
    try:
        surv_func = cox_model.predict_survival_function(df)
        fig, ax = plt.subplots()
        surv_func.plot(ax=ax)
        ax.set_title('Predicted Survival Curve')
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        plot_str = fig_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        return f"Error generating survival curve: {e}", 500

    # Compute concordance index
    c_index = getattr(cox_model, 'concordance_index_', None)

    # Compute risk score and category
    try:
        raw_percentile = cox_model.predict_percentile(df) / 100.0
        risk_score = abs(float(raw_percentile))
        bins = [0, 0.33, 0.66, 1]
        labels = ['Low', 'Medium', 'High']
        risk_cat = pd.cut([risk_score], bins=bins, labels=labels)[0]
    except Exception:
        risk_score = None
        risk_cat = None

    # Build JSON response
    response = {
        'plot': plot_str,
        'c_index': c_index,
        'risk_category': str(risk_cat)
    }
    return jsonify(response)

@app.route('/detection', methods=['POST'])
def detection():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, 'upload.zip')
            file.save(zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            slices = load_dicom_images(temp_dir)
            if not slices:
                return jsonify({"error": "No DICOM files found"}), 400

            volume, _ = convert_to_3d_volume(slices)
            volume_processed = normalize_hu(fast_resize(volume, (64,128,128)))

            input_tensor = torch.tensor(volume_processed).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                pred_t = torch.argmax(model_t(input_tensor), dim=1).item()
                pred_n = torch.argmax(model_n(input_tensor), dim=1).item()
                pred_m = torch.argmax(model_m(input_tensor), dim=1).item()
                pred_loc = torch.argmax(model_loc(input_tensor), dim=1).item()

            # Map numeric predictions to human-readable labels
            T_STAGE_CLASSES = {
                0: "T1a",
                1: "T1b",
                2: "T2a",
                3: "T2b",
                4: "T3",
                5: "T4",
                6: "Tis"
            }

            N_STAGE_CLASSES = {
                0: "N0",
                1: "N1",
                2: "N2"
            }

            M_STAGE_CLASSES = {
                0: "M0",
                1: "M1a",
                2: "M1b"
            }

            LOCATION_CLASSES = {
                0: "L Lingula",
                1: "LLL",
                2: "LUL",
                3: "RLL",
                4: "RML",
                5: "RUL"
            }

            pred_t_label = T_STAGE_CLASSES.get(pred_t, "Unknown")
            pred_n_label = N_STAGE_CLASSES.get(pred_n, "Unknown")
            pred_m_label = M_STAGE_CLASSES.get(pred_m, "Unknown")
            pred_loc_label = LOCATION_CLASSES.get(pred_loc, "Unknown")

            return jsonify({
                "T": pred_t_label,
                "N": pred_n_label,
                "M": pred_m_label,
                "Location": pred_loc_label
            })

    except Exception as e:
        import traceback
        print(f"Error in /detection: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/complications', methods=['POST'])
def complications():
    try:
        data = request.json
        
        # Map inputs using mappings
        for key in data:
            if key in mappings:
                data[key] = map_input(key, data[key])
        
        # Convert to DataFrame for model input
        input_df = pd.DataFrame([data]).astype(float)
        
        # Predict
        pred_catl = model_ctype_catl.predict(input_df)[0]
        pred_ctypel = model_ctypel.predict(input_df)[0]
        pred_gap = model_comp_gap_category.predict(input_df)[0]
        
        # Prediction probabilities for ctypel
        pred_ctypel_proba = model_ctypel.predict_proba(input_df)[0]
        top_5_indices = np.argsort(pred_ctypel_proba)[-5:][::-1]
        
        top_5 = [{"complication": inv_ctypel_mapping[i], "probability": float(pred_ctypel_proba[i])} for i in top_5_indices]
        
        response = {
            "severity": ctype_catl_mapping.get(pred_catl, "Unknown"),
            "complication_type": inv_ctypel_mapping.get(pred_ctypel, "Unknown"),
            "treatment_timing": comp_gap_category_mapping.get(pred_gap, "Unknown"),
            "top_5_complications": top_5
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/recurrence', methods=['POST'])
def recurrence():
    """
    Endpoint for recurrence prediction.
    """
    try:
        # Load recurrence models on demand to avoid loading them at startup
        coxph_model = joblib.load("models/coxph_model.pkl")
        tumour_event_model = joblib.load("models/tumor_event_prediction_model_balanced_rf.pkl")
        pfi_binary_model = joblib.load("models/pfi_ensemble_model.pkl")
        age_scaler = joblib.load("models/age_scaler.pkl")
        feature_scaler = joblib.load("models/feature_scaler.pkl")
        label_encoder = joblib.load("models/label_encoders.pkl")
        
        # Get input data from request
        input_data = request.json
        
        # Process data - create DataFrame from JSON
        input_df = pd.DataFrame([input_data])
        
        # Scale Age
        input_df["age"] = input_df["Age"]
        input_df["age_scaled"] = age_scaler.transform(input_df[["age"]])
        
        # Encode Gender
        input_df["is_female"] = input_df["Gender"].apply(lambda x: 1 if x == "Female" else 0)
        
        # Process categorical columns
        categorical_columns = ["ajcc_pathologic_tumor_stage", "treatment_outcome_first_course"]
        for col in categorical_columns:
            input_df[col] = input_df[col].replace({
                'nan': 'nan', 
                '[Discrepancy]': '[Discrepancy]', 
                '[Not Evaluated]': '[Not Evaluated]',
                'Unknown': '[Unknown]', 
                '[Unknown]': '[Unknown]'
            })
            encoded_col = f"{col}_encoded"
            input_df[encoded_col] = label_encoder[col].transform(input_df[col])
        
        # Get gene columns
        gene_columns = [col for col in input_df.columns if col.endswith('_rnaseq')]
        
        # Define feature order
        feature_order = ["age_scaled", "is_female"] + gene_columns + ["ajcc_pathologic_tumor_stage_encoded", "treatment_outcome_first_course_encoded"]
        
        # Scale features
        input_df[feature_order] = feature_scaler.transform(input_df[feature_order])
        
        # --- PFI Binary Prediction ---
        pfi_binary_features = feature_order  # Same features
        input_df_pfi_binary = input_df[pfi_binary_features]
        pfi_prob = pfi_binary_model.predict_proba(input_df_pfi_binary)[0][1]
        pfi_pred = "Yes" if pfi_prob > 0.5 else "No"
        
        # --- Tumor Event Prediction ---
        input_df_tumour = input_df[feature_order]  # Same features again
        
        if pfi_pred == "No":  # If no progression predicted
            pred_label = "No New Tumor"
            prob_dict = {"No New Tumor": 1.0}
            
            # Use default encoding for "no event"
            event_pred = 0  # Default fallback
            if "None" in label_encoder['new_tumor_event_type'].classes_:
                event_pred = label_encoder['new_tumor_event_type'].transform(["None"])[0]
            elif "No Recurrence" in label_encoder['new_tumor_event_type'].classes_:
                event_pred = label_encoder['new_tumor_event_type'].transform(["No Recurrence"])[0]
        else:  # If progression predicted
            event_prob = tumour_event_model.predict_proba(input_df_tumour)[0]
            event_pred = np.argmax(event_prob)
            event_encoder = label_encoder['new_tumor_event_type']
            
            # Check if prediction is valid
            if event_pred >= len(event_encoder.classes_):
                pred_label = "Unknown (Prediction Error)"
                prob_dict = {str(i): float(prob) for i, prob in enumerate(event_prob)}
            else:
                pred_label = event_encoder.inverse_transform([event_pred])[0]
                class_labels = event_encoder.classes_
                prob_dict = {str(class_labels[i]): float(prob) for i, prob in enumerate(event_prob)}
        
        # --- Feature Importance ---
        feature_importance_data = None
        if hasattr(tumour_event_model, 'feature_importances_') and pfi_pred == "Yes":
            feature_importances = tumour_event_model.feature_importances_
            top_10_features = pd.DataFrame({
                'Feature': feature_order,
                'Importance': feature_importances.tolist()
            }).sort_values(by='Importance', ascending=False).head(10)
            
            feature_importance_data = {
                'features': top_10_features['Feature'].tolist(),
                'importances': top_10_features['Importance'].tolist()
            }
        
        # --- PFI Time (CoxPH Survival Curve) ---
        coxph_features = feature_order + ["new_tumor_event_type_encoded"]
        input_df_coxph = input_df.copy()
        input_df_coxph["new_tumor_event_type_encoded"] = event_pred
        input_df_coxph = input_df_coxph[coxph_features]
        
        max_pfi_training = 1200  # Adjust based on your training data
        times = np.arange(0, max(731, max_pfi_training + 1), step=1)
        survival_function = coxph_model.predict_survival_function(input_df_coxph, times=times)
        
        # Calculate key PFI metrics
        try:
            median_pfi_time = survival_function.index[np.where(survival_function.values < 0.5)[0][0]]
            median_pfi_numeric = float(median_pfi_time)
        except IndexError:
            median_pfi_time = None
            median_pfi_numeric = None
        
        one_year_pfi = float(survival_function.loc[365].iloc[0])
        two_year_pfi = float(survival_function.loc[730].iloc[0])
        
        # Risk level determination
        if pfi_prob > 0.75 or (median_pfi_numeric and median_pfi_numeric < 365):
            risk_level = "High"
        elif pfi_prob > 0.25 or (median_pfi_numeric and median_pfi_numeric < 730):
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Top features from CoxPH model
        coxph_summary = coxph_model.summary[['coef']]
        coxph_summary['Absolute Coefficient'] = coxph_summary['coef'].abs()
        top_10_features_coxph = coxph_summary.sort_values(by='Absolute Coefficient', ascending=False).head(10)
        
        # Generate plots
        plots = {}
        
        # Survival curve
        fig, ax = plt.subplots(figsize=(10, 6))
        survival_function.plot(ax=ax, color="#2e86c1", linewidth=2.5)
        ax.axhline(0.5, color='#e74c3c', linestyle='--', alpha=0.7, label='Median (50%)')
        ax.axvline(365, color='#27ae60', linestyle=':', alpha=0.7, label='1 Year')
        ax.axvline(730, color='#f1c40f', linestyle=':', alpha=0.7, label='2 Years')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title("PFI Survival Curve", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (Days)", fontsize=12)
        ax.set_ylabel("Progression-Free Probability", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend()
        plots['survival_curve'] = fig_to_base64(fig)
        plt.close(fig)
        
        # Feature importance plot for CoxPH model
        fig, ax = plt.subplots(figsize=(10, 6))
        features = top_10_features_coxph.index.tolist()
        coeffs = top_10_features_coxph['coef'].tolist()
        colors = ['red' if coef > 0 else 'blue' for coef in coeffs]
        ax.barh(features, coeffs, color=colors)
        ax.set_xlabel("Coefficient Value", fontsize=12)
        ax.set_ylabel("Feature Name", fontsize=12)
        ax.set_title("Top Features - Cox Model", fontsize=14)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.invert_yaxis()
        plots['coxph_features'] = fig_to_base64(fig)
        plt.close(fig)
        
        # Feature importance for tumor event if available
        if feature_importance_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            features = feature_importance_data['features'][:10]  # Top 10
            importances = feature_importance_data['importances'][:10]
            sns.barplot(x=importances, y=features, palette="coolwarm", ax=ax)
            ax.set_title("Top Important Features (Tumor Event Prediction)")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature Name")
            plots['tumor_features'] = fig_to_base64(fig)
            plt.close(fig)
        
        # Compile results
        results = {
            'pfi_probability': float(pfi_prob),
            'pfi_prediction': pfi_pred,
            'event_prediction': pred_label,
            'event_probabilities': prob_dict,
            'median_pfi_time': median_pfi_numeric,
            'one_year_pfi': one_year_pfi,
            'two_year_pfi': two_year_pfi,
            'risk_level': risk_level,
            'top_features_coxph': {
                'features': top_10_features_coxph.index.tolist(),
                'coefficients': top_10_features_coxph['coef'].tolist(),
                'abs_coefficients': top_10_features_coxph['Absolute Coefficient'].tolist()
            }
        }
        
        if feature_importance_data:
            results['top_features_tumor'] = feature_importance_data
        
        response = {
            'results': results,
            'plots': plots
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
