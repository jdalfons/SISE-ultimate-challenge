import streamlit as st
import datetime
import os

DIRECTORY = "audios"
FILE_NAME = "audio.wav"

def application():
    st.title("SISE ultimate challenge")
    st.write("C'est le dernier challenge de la formation SISE.")
    st.markdown("""
        **Overview:**
        - Analyse de logs
        - Analyse de données
        - Machine learning
    """)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Record Audio", "Register Audio"])

    with tab1:
        st.header("Record Audio")
        st.write("Here you can record audio.")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
        if audio_file is not None:
            
            with open(f"audios/{FILE_NAME}", "wb") as f:
                f.write(audio_file.getbuffer())
                st.success(f"Saved file: {FILE_NAME}")

    with tab2:
        st.header("Register Audio")
        st.write("Here you can register audio.")
        

    file = os.path.join(DIRECTORY, FILE_NAME)
    if os.path.exists(file):
        st.markdown("## File registered:")
        audio_data = st.audio(file, format='audio/wav', start_time=0)
