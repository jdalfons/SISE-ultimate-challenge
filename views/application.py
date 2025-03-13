import streamlit as st
from st_audiorec import st_audiorec
import datetime
import os
from src.model.transcriber import transcribe_audio

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
    
    tab1, tab2, tab3 = st.tabs(["⬆️ Record Audio", "🔈 Realtime Audio", "📝 Transcription"])

    with tab1:
        st.header("⬆️ Upload Audio Record")
        st.write("Here you can upload a pre-recorded audio.")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
        if audio_file is not None:
            
            with open(f"{DIRECTORY}/{FILE_NAME}", "wb") as f:
                f.write(audio_file.getbuffer())
                st.success(f"Saved file: {FILE_NAME}")

    with tab2:
        st.header("🔈 Realtime Audio Record")
        st.write("Here you can record an audio.")
        
        if st.button("Register", key="register-button"):
            st.success("Audio registered successfully.")

        audio_file = st_audiorec()

        if audio_file is not None:
            st.audio(audio_file, format='audio/wav')
            
    with tab3:
        st.header("📝 Speech2Text Transcription")
        st.write("Here you can get the audio transcript.")

        save = st.checkbox("Save transcription to .txt", value=False, key="save-transcript")

        ############################# A décommenté quand ce sera débogué
        if st.button("Transcribe", key="transcribe-button"):
        #     # Fonction pour transcrire l'audio
        #     transcription = transcribe_audio(st.audio)

        #     # Charger et transcrire l'audio
        #     # audio, rate = load_audio(audio_file_path) # (re)chargement de l'audio si nécessaire
        #     transcription = transcribe_audio(audio_file, sampling_rate=16000)

        #     # Afficher la transcription
        #     st.write("Transcription :", transcription)
                

             st.success("Audio registered successfully.")
        #     if save:
        #         file_path = "transcript.txt"
    
        #         # Write the text to the file
        #         with open(file_path, "w") as file:
        #             file.write(transcription)
                
        #         st.success(f"Text saved to {file_path}")

            