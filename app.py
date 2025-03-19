import os
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import pydicom
import torch
import torch.nn.functional as F
import tempfile
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from glob import glob
from torchvision.models.video import r2plus1d_18
from sklearn.preprocessing import MinMaxScaler
from lifelines import CoxPHFitter
from models.resnet import resnet50

# ✅ Load Models with Caching
@st.cache_resource(ttl=3600)
def load_models():
    cox_model = joblib.load("cox_model.pkl")
    tumour_event_model = joblib.load("tumor_event_prediction_model_balanced_rf.pkl")
    pfi_binary_model = joblib.load("pfi_ensemble_model.pkl")
    age_scaler = joblib.load("age_scaler.pkl")
    feature_scaler = joblib.load("feature_scaler.pkl")
    return cox_model, tumour_event_model, pfi_binary_model, age_scaler, feature_scaler

cox_model, tumour_event_model, pfi_binary_model, age_scaler, feature_scaler = load_models()

# ✅ Load PyTorch TNM Models
@st.cache_resource
def load_pytorch_model(model_path, num_classes):
    model = resnet50(sample_input_D=64, sample_input_H=128, sample_input_W=128, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_r2plus1d_model(model_path, num_classes):
    model = r2plus1d_18(weights=None)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.7), torch.nn.Linear(model.fc.in_features, num_classes))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=False)
    model.eval()
    return model

N_model = load_pytorch_model("trained_modelN.pth", num_classes=3)
M_model = load_pytorch_model("trained_modelM.pth", num_classes=3)
T_stage_model = load_r2plus1d_model("model_TStage.pth", num_classes=7)
Tumor_model = load_r2plus1d_model("model_TumorrLocation.pth", num_classes=6)

# ✅ Class Labels
N_CLASS_LABELS = {0: "N0", 1: "N2", 2: "N1"}
M_CLASS_LABELS = {0: "M0", 1: "M1b", 2: "M1a"}
T_STAGE_LABELS = {0: "T3", 1: "T1b", 2: "T2a", 3: "T1a", 4: "Tis", 5: "T2b", 6: "T4"}
TUMOR_LOCATION_LABELS = {0: 'RUL', 1: 'RML', 2: 'LUL', 3: 'RLL', 4: 'LLL', 5: 'L Lingula'}
# Sidebar for navigation
page = st.sidebar.selectbox("Select a page", options=["Prognosis", "Detection", "Complications", "Recurrence"])

# --- Prognosis Page (unchanged) ---
if page == "Prognosis":
    st.title("Survival Prediction App")
    st.write("Upload a single row of patient data to get a survival curve.")


    scaler = MinMaxScaler(feature_range=(0, 1))

    uploaded_file = st.file_uploader("Upload a CSV file with one row", type=["csv"])
    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        if input_data.shape[0] != 1:
            st.error("Please upload exactly one row of data.")
        else:
            expected_columns = cox_model.params_.index.tolist()
            missing_cols = [col for col in expected_columns if col not in input_data.columns]

            # Get the unique values from the list
            unique_columns = set(expected_columns)
 
            # Get the number of unique values
            num_unique_columns = len(unique_columns)

            # Print the result
            print(f"Number of unique columns: {num_unique_columns}")
            input_data = input_data[cox_model.params_.index.tolist()]

            if missing_cols:
                st.error(f"Uploaded file is missing required columns: {', '.join(missing_cols)}")
            else:
                try:

                    survival_function = cox_model.predict_survival_function(input_data)

                    # Predict survival function
                    print("I am here") 
                    survival_function = cox_model.predict_survival_function(input_data)
                    print("I am here2") 
                    fig, ax = plt.subplots()
                    survival_function.plot(ax=ax, color="blue", linewidth=2)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.set_title("Predicted Survival Curve", fontsize=14, fontweight="bold")
                    ax.set_xlabel("Time (months)", fontsize=12)
                    ax.set_ylabel("Survival Probability", fontsize=12)
                    st.pyplot(fig)

                    # Compute and display Concordance Index (C-index)
                    
                    c_index = cox_model.concordance_index_
                    # st.subheader(f"Model Concordance Index (C-index): {c_index:.3f}")

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


                    st.subheader("Top 5 Positive & Negative Features")
                    feature_importance = cox_model.params_
                    top_positive = feature_importance.nlargest(5)
                    top_negative = feature_importance.nsmallest(5)
                    top_features_combined = pd.concat([top_positive, top_negative])
                    colors = ['red' if coef > 0 else 'blue' for coef in top_features_combined]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(top_features_combined.index, top_features_combined, color=colors)
                    ax.set_xlabel("Coefficient Value", fontsize=12)
                    ax.set_ylabel("Feature Name", fontsize=12)
                    ax.set_title("Top 5 Positive & Negative Features - Cox Model", fontsize=14)
                    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    ax.invert_yaxis()
                    st.pyplot(fig)

                    # Compute Individual Risk Score


                    # Load historical patient data for comparison (if available)
                    if patients_data is not None:
                        historical_risk_scores = (patients_data * cox_model.params_).sum(axis=1)
                        
                        historical_risk_scores_scaled = scaler.fit_transform(historical_risk_scores.values.reshape(-1, 1)).flatten()
                        # risk_score = (input_data * cox_model.params_).sum(axis=1).values[0]
                        risk_score = cox_model.predict_percentile(input_data)/100
                        if risk_score < 0:
                            risk_score=risk_score*-1
                        print(risk_score)

                        # st.subheader(f"Patient-Specific Risk Score: {risk_score:.3f}")

                        min_risk_score = min(0, risk_score)  # Ensure no negative values
                        max_risk_score = max(1, risk_score)  # Ensure no risk score exceeds 1
                        bin_edges = [min_risk_score, 0.33, 0.66, max_risk_score]



                        #Categorize Risk based on these dynamic bins
                        risk_category = pd.cut([risk_score], bins=bin_edges, labels=["Low", "Medium", "High"], include_lowest=True)[0]
                        # st.subheader(f"Risk Category: {risk_category} Risk")
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cox_model.plot(columns=top_features_combined.index, ax=ax,hazard_ratios=True)
                        
                        # Set title and labels
                        ax.set_title("Top 5 Features - Cox Proportional Hazards Model")
                        ax.set_ylabel("Hazard Ratios")
                        st.pyplot(fig)




                        # # Plot Risk Score Distribution
                        # st.subheader("Risk Score Comparison with Historical Patients")
                        # fig, ax = plt.subplots()
                        # sns.histplot(historical_risk_scores_scaled, bins=30, kde=True, color="gray")
                        # ax.axvline(risk_score, color="red", linestyle="dashed", linewidth=2, label="This Patient")
                        # ax.set_title("Risk Score Distribution")
                        # ax.set_xlabel("Risk Score (Scaled)")
                        # ax.set_ylabel("Frequency")
                        # ax.set_xlim(0, 1)  # Set x-axis range
                        # ax.legend()
                        # st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# --- Detection Page (unchanged) ---
elif page == "Detection":
    st.title("Lung Cancer Staging and Tumor Location Prediction")
    uploaded_folder = st.file_uploader("Upload DICOM files", accept_multiple_files=True, type=["dcm"])
    if uploaded_folder:
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_folder:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

        slices = [pydicom.dcmread(f) for f in glob(os.path.join(temp_dir, "*.dcm"))[:50]]
        slices.sort(key=lambda x: float(x.SliceLocation))
        volume = np.array([s.pixel_array for s in slices])

        input_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_tensor_r2plus1d = input_tensor.expand(-1, 3, -1, -1, -1)

        predicted_N_label, confidence_N = N_CLASS_LABELS[torch.argmax(N_model(input_tensor))], F.softmax(N_model(input_tensor), dim=1).max().item()
        predicted_M_label, confidence_M = M_CLASS_LABELS[torch.argmax(M_model(input_tensor))], F.softmax(M_model(input_tensor), dim=1).max().item()
        predicted_T_stage, confidence_T = T_STAGE_LABELS[torch.argmax(T_stage_model(input_tensor_r2plus1d))], F.softmax(T_stage_model(input_tensor_r2plus1d), dim=1).max().item()
        predicted_tumor_location, confidence_TL = TUMOR_LOCATION_LABELS[torch.argmax(Tumor_model(input_tensor_r2plus1d))], F.softmax(Tumor_model(input_tensor_r2plus1d), dim=1).max().item()

        st.subheader("Predicted Stages and Tumor Location")
        st.write(f"**T Stage:** {predicted_T_stage} (Confidence: {confidence_T:.2f})")
        st.write(f"**N Stage:** {predicted_N_label} (Confidence: {confidence_N:.2f})")
        st.write(f"**M Stage:** {predicted_M_label} (Confidence: {confidence_M:.2f})")
        st.write(f"**Tumor Location:** {predicted_tumor_location} (Confidence: {confidence_TL:.2f})")

# --- Complications Page (unchanged) ---
elif page == "Complications":
    st.title("Complications")
    st.write("Content related to complications goes here.")

# --- Recurrence Page ---
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
        "treatment_outcome_first_course": [ 'Complete Remission/Response', 'Progressive Disease',
                                           'Partial Remission/Response', 'Stable Disease', '[Not Evaluated]']
    }
    # Define encoded column names and model-specific feature orders
    encoded_columns = ["is_female", "ajcc_pathologic_tumor_stage_encoded", "treatment_outcome_first_course_encoded"]
    all_columns = ["Age"] + categorical_columns + gene_columns
    
    # Define feature orders for each model
    tumour_event_features = ["age_scaled", "is_female"] + gene_columns + ["ajcc_pathologic_tumor_stage_encoded", "treatment_outcome_first_course_encoded"]
    pfi_binary_features = ["age_scaled", "is_female"] + gene_columns + ["ajcc_pathologic_tumor_stage_encoded", "treatment_outcome_first_course_encoded"]
    coxph_features = ["age_scaled", "is_female"] + gene_columns + ["ajcc_pathologic_tumor_stage_encoded", "treatment_outcome_first_course_encoded", "new_tumor_event_type_encoded"]

    # Initialize session state for input data
    if "input_data" not in st.session_state:
        st.session_state.input_data = pd.DataFrame(columns=all_columns)

    # Random value generator
    def generate_random_values():
        random_data = {}
        random_data["Age"] = np.random.randint(18, 90)
        random_data["Gender"] = np.random.choice(categorical_options["Gender"])
        for col in categorical_columns[1:]:  # Skip Gender
            options = [opt if opt != 'nan' else 'Unknown' for opt in categorical_options[col]]
            random_data[col] = np.random.choice(options)
        for col in gene_columns:
            min_val, max_val = gene_ranges[col]
            random_data[col] = np.random.uniform(min_val, max_val)
        st.session_state.input_data = pd.DataFrame([random_data], columns=all_columns)

    # Button to generate random values
    if st.button("Generate Random Values"):
        generate_random_values()

    # Input form
    with st.form(key="recurrence_form"):
        st.subheader("Patient Data Input")
        input_data = {}
        
        # Age input
        input_data["Age"] = st.number_input("Age", min_value=18, max_value=100, value=int(st.session_state.input_data["Age"].iloc[0]) if not st.session_state.input_data.empty else 50)

        # Categorical inputs
        for col in categorical_columns:
            input_data[col] = st.selectbox(
                col, 
                options=categorical_options[col], 
                index=categorical_options[col].index(st.session_state.input_data[col].iloc[0]) if not st.session_state.input_data.empty and col in st.session_state.input_data else 0
            )

        # Gene inputs
        for col in gene_columns:
            min_val, max_val = gene_ranges[col]
            input_data[col] = st.number_input(
                col, 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=float(st.session_state.input_data[col].iloc[0]) if not st.session_state.input_data.empty else (min_val + max_val) / 2,
                step=0.01
            )

        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        # Prepare input data with original column names
        input_df = pd.DataFrame([input_data], columns=all_columns)

        # Scale Age and rename to age_scaled
        input_df["age"] = input_df["Age"]
        input_df["age_scaled"] = age_scaler.transform(input_df[["age"]])

        # Encode Gender as is_female (1 for Female, 0 for Male)
        input_df["is_female"] = input_df["Gender"].apply(lambda x: 1 if x == "Female" else 0)

        # Encode other categorical columns
        for col in categorical_columns[1:]:  # Skip Gender
            input_df[col] = input_df[col].replace({'nan': 'nan', '[Discrepancy]': '[Discrepancy]', '[Not Evaluated]': '[Not Evaluated]','Unknown': '[Unknown]', '[Unknown]': '[Unknown]' })
            encoded_col = f"{col}_encoded"
            input_df[encoded_col] = label_encoder[col].transform(input_df[col])

        # Define the exact feature order as per your earlier specification
        feature_order = [
            "age_scaled", "is_female", 
            "ACSL3_rnaseq", "ADAM10_rnaseq", "ADM_rnaseq", "AKT2_rnaseq", "ANPEP_rnaseq", "ATIC_rnaseq", 
            "AURKA_rnaseq", "AURKB_rnaseq", "BCL2_rnaseq", "BIRC3_rnaseq", "BLM_rnaseq", "BMP1_rnaseq", 
            "BMP2K_rnaseq", "BMP6_rnaseq", "BMP8A_rnaseq", "BRCA1_rnaseq", "BRD2_rnaseq", "BRIP1_rnaseq", 
            "BSG_rnaseq", "BTK_rnaseq", "BUB1B_rnaseq", "BUB1_rnaseq", "CAMK1D_rnaseq", "CBFA2T3_rnaseq", 
            "CBLC_rnaseq", "CCL14_rnaseq", "CCL20_rnaseq", "CCL7_rnaseq", "CCR2_rnaseq", "CCR6_rnaseq", 
            "CCR7_rnaseq", "CD109_rnaseq", "CD163_rnaseq", "CD276_rnaseq", "CD300A_rnaseq", "CD302_rnaseq", 
            "CD40LG_rnaseq", "CD48_rnaseq", "CD5_rnaseq", "CD79B_rnaseq", "CDCP1_rnaseq", "CDK16_rnaseq", 
            "CDK19_rnaseq", "CDK1_rnaseq", "CDK4_rnaseq", "CDK6_rnaseq", "CDKL2_rnaseq", "CDKN2A_rnaseq", 
            "CEBPA_rnaseq", "CHEK1_rnaseq", "CIT_rnaseq", "CKLF_rnaseq", "CLEC10A_rnaseq", "CLTCL1_rnaseq", 
            "CMTM3_rnaseq", "COL1A1_rnaseq", "CR2_rnaseq", "CSF2RB_rnaseq", "CTF1_rnaseq", "CTSG_rnaseq", 
            "CXCL17_rnaseq", "DAPK2_rnaseq", "DDX10_rnaseq", "DEFB1_rnaseq", "DKK1_rnaseq", "EIF2AK2_rnaseq", 
            "EPHB2_rnaseq", "EPHB3_rnaseq", "EPO_rnaseq", "EREG_rnaseq", "EWSR1_rnaseq", "EXT1_rnaseq", 
            "FAM3C_rnaseq", "FANCD2_rnaseq", "FIP1L1_rnaseq", "FSTL3_rnaseq", "FUT4_rnaseq", "GDF10_rnaseq", 
            "GMFB_rnaseq", "GMPS_rnaseq", "GNAS_rnaseq", "GPI_rnaseq", "HERPUD1_rnaseq", "HLF_rnaseq", 
            "HMGA1_rnaseq", "HMMR_rnaseq", "HSP90AA1_rnaseq", "IGF1R_rnaseq", "IKZF1_rnaseq", "IL16_rnaseq", 
            "IL1R2_rnaseq", "IL1RN_rnaseq", "IL23A_rnaseq", "IL2RA_rnaseq", "IL32_rnaseq", "IL33_rnaseq", 
            "IL36RN_rnaseq", "IRF4_rnaseq", "ITGA4_rnaseq", "ITGA5_rnaseq", "ITGA6_rnaseq", "ITGAV_rnaseq", 
            "ITGB1_rnaseq", "ITGB3_rnaseq", "ITGB4_rnaseq", "JAG1_rnaseq", "KRAS_rnaseq", "L1CAM_rnaseq", 
            "LASP1_rnaseq", "LY9_rnaseq", "MAP2K1_rnaseq", "MAP3K12_rnaseq", "MAP4K4_rnaseq", "MAPK12_rnaseq", 
            "MAPK4_rnaseq", "MAPK6_rnaseq", "MAPKAPK2_rnaseq", "MAPKAPK5_rnaseq", "MASTL_rnaseq", "MCAM_rnaseq", 
            "MDK_rnaseq", "MECOM_rnaseq", "MELK_rnaseq", "MIF_rnaseq", "MLKL_rnaseq", "MRC1_rnaseq", 
            "MS4A1_rnaseq", "MST1R_rnaseq", "MYB_rnaseq", "MYH9_rnaseq", "NACA_rnaseq", "NBN_rnaseq", 
            "NEK2_rnaseq", "NLK_rnaseq", "NMB_rnaseq", "NONO_rnaseq", "NPM1_rnaseq", "NRAS_rnaseq", 
            "NRP1_rnaseq", "NUAK2_rnaseq", "NUMBL_rnaseq", "NUP98_rnaseq", "OSM_rnaseq", "PAK2_rnaseq", 
            "PBK_rnaseq", "PDGFB_rnaseq", "PGF_rnaseq", "PICALM_rnaseq", "PIK3R1_rnaseq", "PKMYT1_rnaseq", 
            "PLAUR_rnaseq", "PLAU_rnaseq", "PLK1_rnaseq", "PLK4_rnaseq", "PML_rnaseq", "PRKCB_rnaseq", 
            "PRKDC_rnaseq", "PTPRC_rnaseq", "PVR_rnaseq", "RBM15_rnaseq", "RECQL4_rnaseq", "RHOH_rnaseq", 
            "RIPK2_rnaseq", "RPN1_rnaseq", "RPS6KB1_rnaseq", "RPS6KL1_rnaseq", "RUNX1_rnaseq", "SCYL1_rnaseq", 
            "SCYL2_rnaseq", "SDHAF2_rnaseq", "SELP_rnaseq", "SEMA3C_rnaseq", "SEMA4B_rnaseq", "SEMA7A_rnaseq", 
            "SH3GL1_rnaseq", "SIGLEC6_rnaseq", "SIGLEC7_rnaseq", "SLAMF1_rnaseq", "SLC3A2_rnaseq", 
            "SLC44A1_rnaseq", "SMO_rnaseq", "SPECC1_rnaseq", "SPP1_rnaseq", "SRPK3_rnaseq", "STC2_rnaseq", 
            "STIL_rnaseq", "STK17A_rnaseq", "STK24_rnaseq", "STK32A_rnaseq", "STK3_rnaseq", "STYK1_rnaseq", 
            "TBK1_rnaseq", "TFG_rnaseq", "TLR10_rnaseq", "TLR2_rnaseq", "TMPRSS2_rnaseq", "TNFRSF10C_rnaseq", 
            "TNFRSF1A_rnaseq", "TNFSF4_rnaseq", "TPM3_rnaseq", "TRIM28_rnaseq", "TTK_rnaseq", "TWF1_rnaseq", 
            "VEGFA_rnaseq", "VEGFC_rnaseq", "VGF_rnaseq", "ZNF384_rnaseq", 
            "ajcc_pathologic_tumor_stage_encoded", "treatment_outcome_first_course_encoded"
        ]

        # Apply feature scaling using the exact feature order
        input_df[feature_order] = feature_scaler.transform(input_df[feature_order])

        # Drop unencoded columns
        input_df = input_df.drop(columns=["Age", "age", "Gender", "ajcc_pathologic_tumor_stage", "treatment_outcome_first_course"])

                # Clinical Notes Dictionary
                # Clinical Notes Dictionary
                # Clinical Notes Dictionary
        CLINICAL_NOTES = {
            "No New Tumor": "No evidence of tumor recurrence detected",
            "Local": "Cancer has returned to the original site",
            "Regional": "Cancer has spread to nearby lymph nodes or tissues",
            "Distant": "Cancer has spread to distant organs (Stage IV)",
            "New Primary": "New distinct cancer type identified",
            "Unknown": "Recurrence type needs further investigation"
        }

        # --- PFI Binary Prediction ---
        input_df_pfi_binary = input_df[pfi_binary_features]
        pfi_prob = pfi_binary_model.predict_proba(input_df_pfi_binary)[0][1]
        pfi_pred = "Yes" if pfi_prob > 0.5 else "No"

        # --- Tumor Event Prediction ---
        input_df_tumour = input_df[tumour_event_features]
        if pfi_pred == "No":  # If PFI binary is "No" (no progression)
            pred_label = "No New Tumor"
            prob_dict = {"No New Tumor": 1.0}  # 100% probability for "No New Tumor"
            # Use a default or existing "no event" label for encoding
            event_pred = 0  # Default fallback; adjust if your encoder has a "None" or similar label
            if "None" in label_encoder['new_tumor_event_type'].classes_:
                event_pred = label_encoder['new_tumor_event_type'].transform(["None"])[0]
            elif "No Recurrence" in label_encoder['new_tumor_event_type'].classes_:
                event_pred = label_encoder['new_tumor_event_type'].transform(["No Recurrence"])[0]
        else:  # If PFI binary is "Yes" (progression predicted)
            event_prob = tumour_event_model.predict_proba(input_df_tumour)[0]
            event_pred = np.argmax(event_prob)  # Scalar integer (e.g., 4)
            event_encoder = label_encoder['new_tumor_event_type']

            # Check if predicted index is valid
            if event_pred >= len(event_encoder.classes_):
                st.error(f"Prediction {event_pred} is out of range for classes {event_encoder.classes_}")
                pred_label = "Unknown (Prediction Error)"
                prob_dict = {str(class_labels[i]): float(prob) for i, prob in enumerate(event_prob)}
            else:
                pred_label = event_encoder.inverse_transform([event_pred])[0]
                class_labels = event_encoder.classes_
                prob_dict = {str(class_labels[i]): float(prob) for i, prob in enumerate(event_prob)}

        # Get top 10 important features for tumor event prediction
        if hasattr(tumour_event_model, 'feature_importances_') and pfi_pred == "Yes":
            feature_importances = tumour_event_model.feature_importances_
            top_10_features_tumour = pd.DataFrame({
                'Feature': tumour_event_features,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False).head(10)
        else:
            top_10_features_tumour = None

        # --- PFI Time (CoxPH Survival Curve) ---
        input_df_coxph = input_df.copy()
        input_df_coxph["new_tumor_event_type_encoded"] = event_pred  # Use the precomputed event_pred
        input_df_coxph = input_df_coxph[coxph_features]
        max_pfi_training = 1200  # Replace with your actual max PFI time from training data
        times = np.arange(0, max(731, max_pfi_training + 1), step=1)
        survival_function = coxph_model.predict_survival_function(input_df_coxph, times=times)

        # Get the maximum time in the survival function
        max_pfi_time = survival_function.index[-1]

        # Calculate key PFI metrics (in days)
        try:
            median_pfi_time = survival_function.index[np.where(survival_function.values < 0.5)[0][0]]
        except IndexError:
            median_pfi_time = "Not reached within available range"
        one_year_pfi = survival_function.loc[365].iloc[0]
        two_year_pfi = survival_function.loc[730].iloc[0]

        # Risk Stratification (based on PFI in days)
        if pfi_prob > 0.75 or (isinstance(median_pfi_time, float) and median_pfi_time < 365):
            risk_level = "High"
            risk_color = "#e74c3c"
        elif pfi_prob > 0.25 or (isinstance(median_pfi_time, float) and median_pfi_time < 730):
            risk_level = "Medium"
            risk_color = "#f1c40f"
        else:
            risk_level = "Low"
            risk_color = "#27ae60"

        # --- Output Dashboard ---
        st.subheader("Clinical Decision Support Dashboard")
        st.markdown(f"### Patient Risk Profile: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tumor Event Type", pred_label)
        with col2:
            st.metric("Progression within 12 Months", pfi_pred, f"{pfi_prob:.1%}")
        with col3:
            st.metric("Median PFI Time", f"{median_pfi_time} days" if isinstance(median_pfi_time, float) else median_pfi_time)

        # Detailed Breakdown
        st.markdown("---")
        st.subheader("Detailed Predictions")

        # Tumor Event Breakdown
        with st.expander("Tumor Event Type Prediction"):
            st.write(f"**Predicted Event:** {pred_label}")
            st.write(f"**Clinical Note:** {CLINICAL_NOTES.get(pred_label, 'Consult oncologist for interpretation')}")
            st.write("**Probability Breakdown:**")
            for label, prob in prob_dict.items():
                st.write(f"- {label}: {prob*100:.2f}%")
            if top_10_features_tumour is not None:
                st.write("**Top 10 Important Features:**")
                st.dataframe(top_10_features_tumour)
            elif pfi_pred == "Yes":
                st.write("Feature importance not available for this model.")

        # PFI Binary Prediction (outside expander)
        st.write(f"**Probability of Progression within 12 Months:** {pfi_prob:.1%}")
        st.write(f"**Clinical Interpretation:** {'High risk of early progression' if pfi_pred == 'Yes' else 'Low risk of early progression'}")

        # Survival Curve with Top 10 Features
        with st.expander("Progression-Free Interval (PFI) Survival Curve"):
            fig, ax = plt.subplots(figsize=(10, 6))
            survival_function.plot(ax=ax, color="#2e86c1", linewidth=2.5)
            ax.axhline(0.5, color='#e74c3c', linestyle='--', alpha=0.7, label='Median (50%)')
            ax.axvline(365, color='#27ae60', linestyle=':', alpha=0.7, label='1 Year (365 days)')
            ax.axvline(730, color='#f1c40f', linestyle=':', alpha=0.7, label='2 Years (730 days)')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_title("Progression-Free Interval (PFI) Survival Curve", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time Until Progression (Days)", fontsize=12)
            ax.set_ylabel("Probability of Remaining Progression-Free", fontsize=12)
            ax.set_ylim(0, 1)
            ax.legend()
            st.pyplot(fig)
            st.write(f"**Median PFI Time:** {median_pfi_time} days" if isinstance(median_pfi_time, float) else f"**Median PFI Time:** {median_pfi_time}")
            st.write(f"**1-Year PFI Rate:** {one_year_pfi:.1%}")
            st.write(f"**2-Year PFI Rate:** {two_year_pfi:.1%}")

            # Top 10 Features for CoxPH Model
            st.write("**Top 10 Influential Features for PFI Time Prediction:**")
            coxph_summary = coxph_model.summary[['coef']]
            coxph_summary['Absolute Coefficient'] = coxph_summary['coef'].abs()
            top_10_features_coxph = coxph_summary.sort_values(by='Absolute Coefficient', ascending=False).head(10)
            top_10_features_coxph = top_10_features_coxph.rename(columns={'coef': 'Coefficient'})
            top_10_features_coxph = top_10_features_coxph[['Coefficient', 'Absolute Coefficient']]
            st.dataframe(top_10_features_coxph)

        # Clinical Recommendations
        st.markdown("---")
        st.subheader("Clinical Recommendations")
        if risk_level == "High":
            st.markdown(f"""
            - **Immediate Action:** High risk of progression within 1 year (median PFI < 365 days). Consider aggressive monitoring or adjuvant therapy.
            - **Diagnostics:** Order imaging (e.g., PET/CT) and molecular profiling.
            - **Therapy:** Discuss targeted therapies or clinical trials.
            """)
        elif risk_level == "Medium":
            st.markdown(f"""
            - **Monitoring:** Moderate risk of progression (median PFI < 730 days). Schedule follow-ups every 3-6 months (90-180 days) with imaging.
            - **Prevention:** Evaluate maintenance therapy options.
            - **Consultation:** Review with multidisciplinary team.
            """)
        else:
            st.markdown(f"""
            - **Routine Care:** Low risk of early progression (median PFI ≥ 730 days). Continue standard follow-up every 6-12 months (180-365 days).
            - **Patient Education:** Reinforce lifestyle modifications.
            - **Reassess:** Repeat assessment if new symptoms arise.
            """)

        # Error Handling
        if max_pfi_time < 730:
            st.warning(f"Survival function truncated at {max_pfi_time:.0f} days, which is less than 730 days. Check training data or model configuration.")