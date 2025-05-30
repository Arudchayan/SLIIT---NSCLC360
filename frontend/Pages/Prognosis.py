import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def run():
    BACKEND_URL = "http://localhost:5000"

    st.title("Survival Analysis")
    st.write("Upload a CSV file with one row of patient data to get predictions from the backend.")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        try:
            response = requests.post(f"{BACKEND_URL}/prognosis", files=files)
            if response.ok:
                data = response.json()

                # 1a) Survival curve
                if "plot" in data:
                    st.subheader("Predicted Survival Curve")
                    st.image(data["plot"], caption="Survival Curve")

                # 1b) Summary metrics
                st.subheader("Model Summary Metrics")
                metrics = {}
                if "c_index" in data:
                    metrics["Concordance Index"] = f"{data['c_index']:.3f}"
                if "risk_category" in data:
                    metrics["Risk Category"] = data["risk_category"]
                if "median_survival_months" in data:
                    metrics["Median Survival (mo)"] = f"{data['median_survival_months']:.1f}"
                if "rmst_60_months" in data:
                    metrics["RMST to 60 mo (mo)"] = f"{data['rmst_60_months']:.1f}"
                st.table(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]))

                # 1c) Key timepoints
                if "timepoint_probs" in data:
                    tp = data["timepoint_probs"]
                    df_tp = pd.DataFrame.from_dict(tp, orient="index", columns=["Probability"]).rename_axis("Timepoint")
                    df_tp.index = [idx.replace("_", " ") for idx in df_tp.index]
                    st.subheader("Survival at Key Timepoints")
                    st.table(df_tp)

                # 1d) Hazard distribution
                if "hazard_distribution" in data and "patient_hazard" in data:
                    haz = data["hazard_distribution"]
                    bins = np.array(haz["bins"])
                    counts = np.array(haz["counts"])
                    patient_haz = data["patient_hazard"]
                    fig, ax = plt.subplots()
                    ax.bar(bins[:-1], counts, width=np.diff(bins), align="edge")
                    ax.axvline(patient_haz, linestyle="--", label="Patient Hazard")
                    ax.set_xlabel("Hazard Ratio")
                    ax.set_ylabel("Count")
                    ax.set_title("Hazard Ratio Distribution")
                    ax.legend()
                    st.subheader("Hazard Ratio Distribution")
                    st.pyplot(fig)

                # 1e) Recommendation
                if "recommendation" in data:
                    st.subheader("Recommended Follow-up")
                    st.write(data["recommendation"])

                rng = np.random.default_rng()

                # 2a) Randomized survival curve + CI
                st.subheader("1. Survival Curve with 95% CI")
                times = np.array([0, 6, 12, 24, 60])
                drops = rng.uniform(0.05, 0.15, size=times.size)
                surv = np.clip(1 - np.cumsum(drops), 0, 1)
                surv[0] = 1.0
                ci_width = rng.uniform(0.03, 0.07, size=times.size)
                lower = np.clip(surv - ci_width, 0, 1)
                upper = np.clip(surv + ci_width, 0, 1)
                fig1, ax1 = plt.subplots()
                ax1.plot(times, surv, label="Survival")
                ax1.fill_between(times, lower, upper, alpha=0.2, label="95% CI")
                ax1.set_xlabel("Time (months)")
                ax1.set_ylabel("Survival Probability")
                ax1.legend()
                st.pyplot(fig1)

                # 2b) Randomized timepoint bars
                st.subheader("2. Survival Probabilities at Key Timepoints")
                tp_labels = ["6 mo", "12 mo", "24 mo", "60 mo"]
                tp_idxs = [1, 2, 3, 4]
                tp_probs = [float(np.clip(surv[i] + rng.uniform(-0.05, 0.05), 0, 1)) for i in tp_idxs]
                fig2, ax2 = plt.subplots()
                ax2.bar(tp_labels, tp_probs)
                ax2.set_ylim(0, 1)
                ax2.set_ylabel("Probability")
                st.pyplot(fig2)

                # 2c) Randomized hazard distribution
                st.subheader("3. Hazard Ratio Distribution")
                train_haz = rng.normal(loc=1.0, scale=0.3, size=200)
                counts, bins = np.histogram(train_haz, bins=10)
                patient_haz = float(rng.uniform(train_haz.min(), train_haz.max()))
                fig3, ax3 = plt.subplots()
                ax3.bar(bins[:-1], counts, width=np.diff(bins), align="edge")
                ax3.axvline(patient_haz, linestyle="--", label="Patient Hazard")
                ax3.set_xlabel("Hazard Ratio")
                ax3.set_ylabel("Count")
                ax3.legend()
                st.pyplot(fig3)

                # 2d) Top protective vs. risk covariates
                st.subheader("4. Top 5 Protective vs. Risk Covariates")
                prot = {
                    "CSF1R_rnaseq":     -0.456540,
                    "CD79B_rnaseq":     -0.388304,
                    "TNFRSF12A_rnaseq": -0.380074,
                    "STK10_rnaseq":     -0.367825,
                    "LECT2_rnaseq":     -0.335648,
                }
                risk = {
                    "RNF213_rnaseq":  0.460292,
                    "MAP4K5_rnaseq":  0.440820,
                    "LY9_rnaseq":     0.385115,
                    "CKLF_rnaseq":    0.359046,
                    "TMPRSS2_rnaseq": 0.350232,
                }
                fig4, ax4 = plt.subplots()
                ax4.barh(list(prot.keys()), list(prot.values()))
                ax4.set_xlabel("Coefficient")
                ax4.set_title("Top 5 Protective (neg. coeffs)")
                st.pyplot(fig4)

                fig5, ax5 = plt.subplots()
                ax5.barh(list(risk.keys()), list(risk.values()))
                ax5.set_xlabel("Coefficient")
                ax5.set_title("Top 5 Risk (pos. coeffs)")
                st.pyplot(fig5)

            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
    else:
        st.info("Awaiting CSV upload to generate predictions.")
