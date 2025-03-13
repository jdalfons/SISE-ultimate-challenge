import streamlit as st
from st_audiorec import st_audiorec

from src.model.transcriber import transcribe_audio


def studio():
    st.title("SISE ultimate challenge")
    st.write("C'est le dernier challenge de la formation SISE.")
    st.markdown("""
        **Overview:**
        - Analyse de logs
        - Analyse de données
        - Machine learning
    """)
    
    st.markdown("---")

    st.header("🎧 Audio File Studio")
    
    tab1, tab2, tab3 = st.tabs(["⬆️ Record Audio", "🔈 Realtime Audio", "📝 Transcription"])

    with tab1:
        st.header("⬆️ Upload Audio Record")
        st.write("Here you can upload a pre-recorded audio.")
        audio_file = st.file_uploader("Upload an audio file", type=["wav"])
        
        if "audio_file" not in st.session_state:
            st.session_state.audio_file = None
        
        if audio_file is not None:
            st.success("Audio file uploaded successfully !")
            st.session_state.audio_file = audio_file
            
            # with open(os.path.join(DIRECTORY,FILE_NAME), "wb") as f:
            #     f.write(audio_file.getbuffer())
            #     st.success(f"Saved file: {FILE_NAME}")

        

    with tab2:
        st.header("🔈 Realtime Audio Record")
        st.write("Here you can record an audio.")

        if "audio_file" not in st.session_state:
            st.session_state.audio_file = None        

        audio_file = st_audiorec()

        if audio_file is not None:
            st.audio(audio_file, format='audio/wav')
            st.success("Audio recorded successfully !")
            st.session_state.audio_file = audio_file

##############################################"realtime audio record"##############################################
        # Boutons pour démarrer et arrêter l'enregistrement
        # start_button = st.button("Démarrer l'enregistrement")
        # stop_button = st.button("Arrêter l'enregistrement")
        # start_stop = st.button("Démarrer/Arrêter l'enregistrement")


        # Zone de visualisation des émotions en temps réel
        # emotion_placeholder = st.empty()
        # final_emotion_placeholder = st.empty()
        # audio = pyaudio.PyAudio()
        # audio_buffer = np.array([])
        # emotion_prediction = "Aucune prédiction"
        # is_recording = False

        # if start_stop:
        #     is_recording = not is_recording                        

        #     # Variables globales pour le partage de données entre threads            
        #     def audio_callback(in_data, frame_count, time_info, status):
        #         global audio_buffer
        #         audio_data = np.frombuffer(in_data, dtype=np.float32)
        #         audio_buffer = np.concatenate((audio_buffer, audio_data))
        #         return (in_data, pyaudio.paContinue)

        #     def predict_emotion_thread():
        #         global audio_buffer, emotion_prediction
        #         while is_recording:
        #             if len(audio_buffer) >= CHUNK:
        #                 chunk = audio_buffer[:CHUNK]
        #                 audio_buffer = audio_buffer[STRIDE:]
        #                 emotion_prediction = predict_emotion(chunk, output_probs=False, sampling_rate=RATE)  # Utilisez votre modèle ici
        #             # time.sleep(0.1)

        #     if is_recording:
        #         audio_buffer = np.array([])
        #         stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, 
        #                         frames_per_buffer=CHUNK, stream_callback=audio_callback)
        #         stream.start_stream()
        #         threading.Thread(target=predict_emotion_thread, daemon=True).start()
        #         st.write("Enregistrement en cours...")
        #     else:
        #         stream.stop_stream()
        #         stream.close()
        #         st.write("Enregistrement arrêté.")

        # emotion_display = st.empty()

        # while is_recording:
        #     emotion_display.write(f"Émotion détectée : {emotion_prediction}")
        #     # time.sleep(0.1)

        # audio.terminate(


            # stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            
            # frames = []
            # real_time_emotions = []

            # while not stop_button:
            #     data = stream.read(CHUNK)
            #     frames.append(data)
                
            #     # Traitement en temps réel (par tranche de 1 seconde)
            #     if len(frames) >= RATE // CHUNK:
            #         audio_segment = np.frombuffer(b''.join(frames[-(RATE // CHUNK):]), dtype=np.int16)
            #         emotion = predict_emotion(audio_segment, output_probs=False, sampling_rate=RATE)
            #         real_time_emotions.append(emotion)
            #         emotion_placeholder.line_chart(real_time_emotions)  # Affichage graphique des émotions

            # # Arrêt de l'enregistrement
            # stream.stop_stream()
            # stream.close()
            # audio.terminate()
            
            # # Sauvegarde de l'audio enregistré
            # wf = wave.open("output.wav", "wb")
            # wf.setnchannels(CHANNELS)
            # wf.setsampwidth(audio.get_sample_size(FORMAT))
            # wf.setframerate(RATE)
            # wf.writeframes(b"".join(frames))
            # wf.close()

            # # Prédiction finale sur tout l'audio enregistré
            # full_audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            # final_emotion = predict_emotion(full_audio_data)
            
            # final_emotion_placeholder.write(f"Émotion finale prédite : {final_emotion}")


##############################################"end realtime audio record"##############################################
            
    with tab3:
        st.header("📝 Speech2Text Transcription")
        st.write("Here you can get the audio transcript.")

        save = st.checkbox("Save transcription to .txt", value=False, key="save-transcript")

        ############################# A décommenté quand ce sera débogué
        if st.button("Transcribe", key="transcribe-button"):
            # Fonction pour transcrire l'audio
            transcription = transcribe_audio(st.audio)

            # Charger et transcrire l'audio
            # audio, rate = load_audio(audio_file_path) # (re)chargement de l'audio si nécessaire
            transcription = transcribe_audio(audio_file, sampling_rate=16000)

            # Afficher la transcription
            st.write("Transcription :", transcription)
                
            st.success("Audio registered successfully.")
            if save:
                file_path = "transcript.txt"
    
                # Write the text to the file
                with open(file_path, "w") as file:
                    file.write(transcription)
                
                st.success(f"Text saved to {file_path}")

            