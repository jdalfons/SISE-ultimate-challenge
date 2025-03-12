import streamlit as st


def about():
    st.title("About")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### About")
        st.write("This dashboard is maintained by the M2 SISE team.")
        st.write("For more information, please visit the [GitHub repository](https://github.com/jdalfons/sise-ultimate-challenge/tree/main).")

    with col2:
        st.markdown("### Collaborators")
        st.write("""
        - [Falonne Kpamegan](https://github.com/marinaKpamegan)
        - [Nancy](https://github.com/yminanc)
        - [Cyril](https://github.com/Cyr-CK)
        - [Juan Alfonso](https://github.com/jdalfons)
        """)