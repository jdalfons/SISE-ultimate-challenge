import streamlit as st
from streamlit_option_menu import option_menu
from views.studio import studio
from views.emotion_analysis import emotion_analysis
from views.about import about
import os
import sys

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath(".")) 

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = None

# Set the logo
st.sidebar.image("img/logo.png", use_container_width=True)

# Create a sidebar with navigation options
# Sidebar navigation with streamlit-option-menu
with st.sidebar:
    # st.image("img/logo.png", use_container_width=True)
    # st.markdown("<h1 style='text-align: center;'>SecureIA Dashboard</h1>", unsafe_allow_html=True)
    # Navigation menu with icons
    selected_tab = option_menu(
        menu_title=None,  # Added menu_title parameter
        options=["Studio", "Emotion Analysis", "About"],
        icons=["record-circle", "robot", "info-circle"],
        menu_icon="cast",
        default_index=0,
        # styles={
        # "container": {"padding": "5px", "background-color": "#f0f2f6"},
        # "icon": {"color": "orange", "font-size": "18px"},
        # "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "black"},
        # "nav-link-selected": {"background-color": "#4CAF50", "color": "white"},
        # }
    )
    
    
if selected_tab == "Studio":
    studio()
elif selected_tab == "Emotion Analysis":
    emotion_analysis()
elif selected_tab == "About":
    about()
        