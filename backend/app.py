from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib
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
    # Placeholder for recurrence endpoint
    return jsonify({'message': 'Recurrence endpoint not implemented yet.'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
