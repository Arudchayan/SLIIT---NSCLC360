import streamlit as st
import importlib

# Define pages and module paths
pages = {
    "Detection": "Pages.Detection",
    "Complications": "Pages.Complication",
    "Prognosis": "Pages.Prognosis",
    "Recurrence": "Pages.Recurrence"
}

# Add a "Home" page key for card UI
all_pages = ["Home"] + list(pages.keys())

# Background colors for home (default) and for pages
background_colors = {
    "Detection": "#143D7D",
    "Complications": "#143D7D",
    "Prognosis": "#143D7D",
    "Recurrence": "#143D7D",
    "Home": "#f4faff",
    None: "#f4faff"
}

# Card content for Home page
card_contents = {
    "Detection": {
        "emoji": "🔍",
        "desc": "Identify the presence of lung cancer using comprehensive biological data to improve early diagnosis and intervention strategies."
    },
    "Complications": {
        "emoji": "⚠️",
        "desc": "Analyze clinical and molecular profiles to anticipate potential complications during and after treatment for better patient management."
    },
    "Prognosis": {
        "emoji": "📈",
        "desc": "Deliver personalized prognostic insights by assessing survival outcomes and disease progression using multi-dimensional data."
    },
    "Recurrence": {
        "emoji": "📄",
        "desc": "Predict the likelihood of lung cancer recurrence to guide long-term monitoring and tailored therapeutic plans."
    }
}

if "page" not in st.session_state:
    st.session_state.page = "Home"

# Inject global CSS and lung background image (only for home page)
def inject_home_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #f4faff;
            transition: background-color 0.5s ease;
            padding: 50px 20px 50px 20px;
        }
        .lung-bg {
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background-image: url('https://cdn.pixabay.com/photo/2020/04/12/10/16/lungs-5030651_1280.png');
            background-repeat: no-repeat;
            background-size: 50%;
            background-position: center;
            opacity: 0.05;
            z-index: -1;
        }
        .container {
            max-width: 1200px;
            margin: auto;
            text-align: center;
            color: #263238;
        }
        h1 {
            font-weight: 800;
            font-size: 3rem;
            color: #143D7D;
            margin-bottom: 10px;
        }
        p.subtitle {
            font-size: 1.25rem;
            margin-bottom: 50px;
            color: #37474F;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }
        .card {
            background-color: #ffffff;
            border-radius: 28px;
            padding: 30px 25px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            transition: all 0.4s ease;
            cursor: pointer;
            user-select: none;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-bottom: 30px;
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
            transition: color 0.4s ease;
        }
        .card p {
            font-size: 1rem;
            color: #455A64;
            line-height: 1.6;
            transition: color 0.4s ease;
            margin-top: auto;
        }
        .card:hover h3, .card:hover p {
            color: #ffffff;
        }
        @media (max-width: 768px) {
            .columns-wrapper {
                flex-direction: column !important;
            }
        }
        div.stButton > button {
            width: 100%;
            height: 100%;
            background: none;
            border: none;
            padding: 0;
            margin: 0;
            font-size: inherit;
            font-family: inherit;
            cursor: pointer;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

# Inject navbar CSS for non-home pages
def inject_navbar_css():
    st.markdown("""
        <style>
        .navbar {
            display: flex;
            justify-content: center;
            background-color: #143D7D;
            padding: 12px 0;
            margin-bottom: 30px;
            border-radius: 8px;
        }
        .navbar select {
            font-size: 1.1rem;
            padding: 6px 12px;
            border-radius: 6px;
            border: none;
            outline: none;
        }
        .navbar select:focus {
            outline: none;
        }
        .stApp {
            font-family: 'Inter', sans-serif;
            transition: background-color 0.5s ease;
            padding: 20px 40px;
            background-color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Render the Home page with cards
def render_home():
    inject_home_css()
    st.markdown('<div class="lung-bg"></div>', unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<h1>NSCLC 360</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Leveraging multiomics data for personalized lung cancer prognosis</p>', unsafe_allow_html=True)

    cols = st.columns(2)
    page_names = list(pages.keys())
    for idx, col in enumerate(cols):
        with col:
            for i in range(idx, len(page_names), 2):
                page_name = page_names[i]
                card_html = f"""
                <div class="card">
                    <h3>{card_contents[page_name]['emoji']} {page_name}</h3>
                    <p>{card_contents[page_name]['desc']}</p>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button(f"select-{page_name}"):
                    st.session_state.page = page_name
                    st.experimental_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Render navbar for other pages
def render_navbar():
    inject_navbar_css()
    # Simple navbar with selectbox to switch pages
    selected = st.selectbox(
        "Navigate to:",
        options=all_pages,
        index=all_pages.index(st.session_state.page) if st.session_state.page in all_pages else 0,
        key="nav_select",
        on_change=lambda: nav_changed()
    )
    # Function to handle navbar change
def nav_changed():
    new_page = st.session_state.nav_select
    if new_page != st.session_state.page:
        st.session_state.page = new_page
        st.experimental_rerun()

# Set background color based on page
bg_color = background_colors.get(st.session_state.page, "#f4faff")
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
    }}
    </style>
""", unsafe_allow_html=True)

# Render UI based on page selection
if st.session_state.page == "Home":
    render_home()
else:
    render_navbar()
    if st.session_state.page in pages:
        try:
            mod = importlib.import_module(pages[st.session_state.page])
            mod.run()
        except Exception as e:
            st.error(f"Error loading page '{st.session_state.page}': {e}")
    else:
        st.write("Select a page from the navigation menu.")
