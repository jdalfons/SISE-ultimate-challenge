import streamlit as st
from streamlit_option_menu import option_menu
from views.home import home
from views.analytics import analytics
from views.ml import ml

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
        options=["Home", "Analytics", "Machine Learning"],
        icons=["house", "bar-chart", "robot"],
        menu_icon="cast",
        default_index=0,
        # styles={
        # "container": {"padding": "5px", "background-color": "#f0f2f6"},
        # "icon": {"color": "orange", "font-size": "18px"},
        # "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "black"},
        # "nav-link-selected": {"background-color": "#4CAF50", "color": "white"},
        # }
    )
    
if selected_tab == "Home":
    home()
elif selected_tab == "Analytics":
    analytics()
elif selected_tab == "Machine Learning":
    ml()
        
# Quick links section after filters and content
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### About")
    st.write("This dashboard is maintained by the M2 SISE team.")
    st.write("For more information, please visit the [GitHub repository](https://github.com/jdalfons/sise-ultimate-challenge/tree/main).")

with col2:
    st.markdown("### Collaborators")
    st.write("""
    - [Warrior 1](https://github.com/jdalfons)
    - [Warrior 2](https://github.com/jdalfons)
    - [Juan Alfonso](https://github.com/jdalfons)
    """)