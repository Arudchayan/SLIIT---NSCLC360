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

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load pre-trained Cox model and optional patient data
# Adjust paths as needed
cox_model = joblib.load('Prognosis/model/cox_model.pkl')
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
    # Placeholder for detection endpoint
    return jsonify({'message': 'Detection endpoint not implemented yet.'})


@app.route('/complications', methods=['POST'])
def complications():
    # Placeholder for complications endpoint
    return jsonify({'message': 'Complications endpoint not implemented yet.'})


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
