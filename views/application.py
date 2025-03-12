import streamlit as st
from st_audiorec import st_audiorec
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
    
    tab1, tab2 = st.tabs(["⬆️ Record Audio", "🔈 Realtime Audio"])

    with tab1:
        st.header("⬆️ Upload Audio")
        st.write("Here you can record audio.")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
        if audio_file is not None:
            
            with open(f"{DIRECTORY}/{FILE_NAME}", "wb") as f:
                f.write(audio_file.getbuffer())
                st.success(f"Saved file: {FILE_NAME}")

    with tab2:
        st.header("🔈 Realtime Audio")
        st.write("Here you can register audio.")
        
        if st.button("Register", key="register-button"):
            st.success("Audio registered successfully.")

        wav_audio_data = st_audiorec()

        if wav_audio_data is not None:
            st.audio(wav_audio_data, format='audio/wav')
            