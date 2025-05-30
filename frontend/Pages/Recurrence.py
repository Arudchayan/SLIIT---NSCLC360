# recurrence.py

import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Import gene ranges and utilities from a shared module if needed
# from shared import gene_ranges, categorical_columns, categorical_options

BACKEND_URL = "http://localhost:5000"

def run():
    st.title("Recurrence Prediction")
    st.write("Predict tumor event type, PFI time, and probability using gene expression data and clinical inputs.")

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

