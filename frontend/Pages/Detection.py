import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Plotting function for lung lobes and tumor location
def plot_lung_with_tumor(tumor_location):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Lung Lobes with Tumor Location Highlighted", fontsize=16)
    ax.axis('off')

    # Draw lung outlines
    left_lung = patches.FancyBboxPatch((0.1, 0.2), 0.35, 0.6,
                                       boxstyle="round,pad=0.05",
                                       edgecolor='navy', facecolor='#d0e7ff', linewidth=2)
    right_lung = patches.FancyBboxPatch((0.55, 0.2), 0.35, 0.6,
                                        boxstyle="round,pad=0.05",
                                        edgecolor='navy', facecolor='#d0e7ff', linewidth=2)
    ax.add_patch(left_lung)
    ax.add_patch(right_lung)

    # Lobe centers
    lobes = {
        "L Lingula": (0.15, 0.55),
        "LLL": (0.27, 0.4),
        "LUL": (0.27, 0.7),
        "RLL": (0.70, 0.4),
        "RML": (0.70, 0.55),
        "RUL": (0.70, 0.7)
    }

    for lobe, (x, y) in lobes.items():
        if lobe == tumor_location:
            color = 'red'
            alpha = 0.9
            size = 0.12
            edge = 'darkred'
        else:
            color = 'gray'
            alpha = 0.3
            size = 0.1
            edge = 'black'

        circle = patches.Circle((x, y), size, facecolor=color, alpha=alpha, edgecolor=edge, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, lobe, fontsize=12, ha='center', va='center', weight='bold', color='black')

    ax.plot([], [], marker='o', color='red', label='Tumor Location', linestyle='None')
    ax.plot([], [], marker='o', color='gray', alpha=0.3, label='Other Lobes', linestyle='None')
    ax.legend(loc='lower center', ncol=2, fontsize=12)
    return fig


def run():
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
                pred_t_label = result.get('T', 'N/A')
                pred_n_label = result.get('N', 'N/A')
                pred_m_label = result.get('M', 'N/A')
                pred_loc_label = result.get('Location', 'N/A')

                st.write(f"T Stage: {pred_t_label}")
                st.write(f"N Stage: {pred_n_label}")
                st.write(f"M Stage: {pred_m_label}")
                st.write(f"Tumor Location: {pred_loc_label}")

                # Explanations
                t_stage_explanation = {
                    "T1a": "The tumor is small and confined.",
                    "T1b": "The tumor is moderate in size but still confined.",
                    "T2a": "Moderate size, grown into nearby lung tissue.",
                    "T2b": "Larger and spread to nearby lung tissue.",
                    "T3": "Large tumor, possibly affecting other organs.",
                    "T4": "Very large tumor invading adjacent structures.",
                    "Tis": "Tumor in situ (localized)."
                }

                n_stage_explanation = {
                    "N0": "No lymph node involvement.",
                    "N1": "Spread to nearby lymph nodes.",
                    "N2": "Spread to more distant lymph nodes.",
                }

                m_stage_explanation = {
                    "M0": "No distant metastasis.",
                    "M1": "Distant metastasis detected.",
                    "M1a": "Less severe distant spread.",
                    "M1b": "Severe distant spread.",
                }

                location_explanation = {
                    "L Lingula": "Left lingula part of the left lung.",
                    "LLL": "Left lower lobe of the left lung.",
                    "LUL": "Left upper lobe of the left lung.",
                    "RLL": "Right lower lobe of the right lung.",
                    "RML": "Right middle lobe of the right lung.",
                    "RUL": "Right upper lobe of the right lung."
                }

                # Show explanations
                st.write(f"**T Stage Explanation:** {t_stage_explanation.get(pred_t_label, 'No explanation available.')}")
                st.write(f"**N Stage Explanation:** {n_stage_explanation.get(pred_n_label, 'No explanation available.')}")
                st.write(f"**M Stage Explanation:** {m_stage_explanation.get(pred_m_label, 'No explanation available.')}")
                st.write(f"**Tumor Location Explanation:** {location_explanation.get(pred_loc_label, 'No explanation available.')}")

                # Tumor visualization
                if pred_loc_label in location_explanation:
                    fig = plot_lung_with_tumor(pred_loc_label)
                    st.pyplot(fig)

                # Summary
                summary_text = (
                    f"**What this means for you:**\n\n"
                    f"Based on your scan, the tumor stage is **{pred_t_label}{pred_n_label}{pred_m_label}**. "
                    f"This suggests {t_stage_explanation.get(pred_t_label, '').lower()} with "
                    f"{n_stage_explanation.get(pred_n_label, '').lower()}, "
                    f"and {m_stage_explanation.get(pred_m_label, '').lower()}.\n\n"
                    f"The tumor is in the **{pred_loc_label}**, which means {location_explanation.get(pred_loc_label, '').lower()}.\n\n"
                    f"Regular check-ups and following your physician’s guidance is recommended."
                )
                st.markdown("---")
                st.header("Summary")
                st.write(summary_text)

            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with backend: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
