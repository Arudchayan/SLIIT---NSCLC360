# Pages/Home.py
import streamlit as st

def run():
    st.markdown("""
        <style>
            * {
                font-family: 'Inter', sans-serif;
            }
            .home-container {
                text-align: center;
                padding: 50px;
                background-color: #f4faff;
                color: #263238;
                position: relative;
            }
            .home-title {
                font-size: 3rem;
                font-weight: 800;
                color: #143D7D;
            }
            .home-subtitle {
                font-size: 1.25rem;
                margin: 20px auto 50px;
                color: #37474F;
                max-width: 700px;
            }
            .card-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
                margin-top: 30px;
            }
            .card {
                background-color: #ffffff;
                border-radius: 28px;
                padding: 30px 25px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                transition: all 0.4s ease;
                cursor: pointer;
                text-align: center;
            }
            .card:hover {
                transform: scale(1.02);
                background-color: #143D7D;
                color: #ffffff;
            }
            .card h3 {
                font-size: 1.5rem;
                margin: 15px 0;
                color: #143D7D;
            }
            .card:hover h3, .card:hover p {
                color: white;
            }
            .card p {
                font-size: 1rem;
                color: #455A64;
                line-height: 1.6;
            }
            .lung-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-image: url('https://cdn.pixabay.com/photo/2020/04/12/10/16/lungs-5030651_1280.png');
                background-repeat: no-repeat;
                background-size: 50%;
                background-position: center;
                opacity: 0.05;
                z-index: -1;
            }
        </style>

        <div class="lung-bg"></div>
        <div class="home-container">
            <div class="home-title">NSCLC 360</div>
            <div class="home-subtitle">Leveraging multiomics data for personalized lung cancer prognosis</div>
        </div>
    """, unsafe_allow_html=True)

    # Columns for navigation
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Detection"):
            st.session_state.selected_page = "Detection"
            st.experimental_rerun()
        st.caption("Identify lung cancer early using biological data.")

    with col2:
        if st.button("⚠️ Complications"):
            st.session_state.selected_page = "Complications"
            st.experimental_rerun()
        st.caption("Predict complications using molecular & clinical features.")

    with col1:
        if st.button("📈 Prognosis"):
            st.session_state.selected_page = "Prognosis"
            st.experimental_rerun()
        st.caption("Predict survival and risk categories.")

    with col2:
        if st.button("📄 Recurrence"):
            st.session_state.selected_page = "Recurrence"
            st.experimental_rerun()
        st.caption("Estimate recurrence and progression-free intervals.")
