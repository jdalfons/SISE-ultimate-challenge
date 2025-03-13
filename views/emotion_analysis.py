import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import librosa
from src.predict import predict_emotion

DIRECTORY = "audios"
FILE_NAME = "audio.wav"
RATE = 16000

def emotion_analysis():
    
    st.header("❤️ Emotion Analysis")

    if st.session_state.audio_file is None:
        st.info("Please, upload or record an audio file in the studio tab")
        st.stop()
    else:
        audio_file = st.session_state.audio_file

        start_inference = st.button("Start emotion recogniton","inf_on_upl_btn")
        emotion_labels = ["colere", "neutre", "joie"]
        colors = ['#f71c1c', '#cac8c8', '#f6d60a']

        if start_inference:
            # Configuration Streamlit
            with st.spinner("Real-time emotion analysis..."):
                # uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["wav", "mp3"])

                if audio_file is not None:
                    # Charger et rééchantillonner l'audio
                    audio, sr = librosa.load(audio_file, sr=RATE)
                    # chunk = audio_file
                    
                    # Paramètres de la fenêtre glissante
                    window_size = 1  # 1 seconde de données
                    hop_length = 0.5  # 0.5 secondes de chevauchement
                    
                    # Créer un graphique en temps réel
                    fig, ax = plt.subplots()
                    lines = [ax.plot([], [], label=emotion)[0] for emotion in emotion_labels]
                    ax.set_ylim(0, 1)
                    ax.set_xlim(0, len(audio) / sr)
                    ax.set_xlabel("Temps (s)")
                    ax.set_ylabel("Probabilité")
                    
                    chart = st.pyplot(fig)
                    
                    scores = [[],[],[]] # 3 émotions pour l'instant
                    
                    # Traitement par fenêtre glissante
                    for i in range(0, len(audio), int(hop_length * sr)):
                        chunk = audio[i:i + int(window_size * sr)]
                        if len(chunk) < int(window_size * sr):
                            break
                        
                        emotion_scores = predict_emotion(chunk, output_probs=True, sampling_rate=RATE)
                        
                        # Mettre à jour le graphique
                        for emotion, line in zip(emotion_labels, lines):
                            xdata = list(line.get_xdata())
                            ydata = list(line.get_ydata())
                            colour = colors[list(emotion_scores).index(emotion)]
                            xdata.append(i / sr)
                            ydata.append(emotion_scores[emotion])
                            scores[list(emotion_scores).index(emotion)].append(emotion_scores[emotion])
                            line.set_data(xdata, ydata)
                            line.set_color(colour)
                        
                        ax.relim()
                        ax.autoscale_view()
                        ax.legend()
                        chart.pyplot(fig, use_container_width=True)
                    
                    # Prepare the styling
                    st.markdown("""        
                                <style>
                                .colored-box {
                                    padding: 10px;
                                    border-radius: 5px;
                                    color: white;
                                    font-weight: bold;
                                    text-align: center;
                                }
                                </style>
                                """
                                , unsafe_allow_html=True)

                    # Dynamically create the specified number of columns
                    columns = st.columns(len(emotion_scores))

                    # emotion_scores_mean = [sum(sublist) / len(sublist) for sublist in scores]
                    emotion_scores_mean = {emotion:sum(sublist) / len(sublist) for emotion, sublist in zip(emotion_labels, scores)}
                    max_emo = max(emotion_scores_mean)
                    emotion_scores_sorted = dict(sorted(emotion_scores_mean.items(), key=lambda x: x[1], reverse=True))
                    colors_sorted = [colors[list(emotion_scores_mean.keys()).index(key)] for key in list(emotion_scores_sorted.keys())]

                    # Add content to each column
                    for i, (col, emotion) in enumerate(zip(columns, emotion_scores_sorted)):
                        color = colors_sorted[i % len(colors_sorted)]  # Cycle through colors if more columns than colors
                        col.markdown(f"""
                                    <div class="colored-box" style="background-color: {color};">
                                        {emotion} : {100*emotion_scores_sorted[emotion]:.2f} %
                                    </div>
                                    """
                        , unsafe_allow_html=True)
                        


                    st.success("Analyse terminée !")
                else:
                    st.warning("You need to load an audio file !")
            
        if start_inference:

            st.subheader("Feedback")

            # Initialisation du fichier CSV
            csv_file = os.path.join("src","predictions","feedback.csv")

            # Vérifier si le fichier CSV existe, sinon le créer avec des colonnes appropriées
            if not os.path.exists(csv_file):
                df = pd.DataFrame(columns=["filepath", "prediction", "feedback"])
                df.to_csv(csv_file, index=False)

            # Charger les données existantes du CSV
            df = pd.read_csv(csv_file)

            with st.form("feedback_form"):
                st.write("What should have been the correct prediction ? (*Choose the same emotion if the prediction was correct*).")
                feedback = st.selectbox("Your answer :", ['Sadness','Anger', 'Disgust', 'Fear', 'Surprise', 'Joy', 'Neutral'])
                submit_button = st.form_submit_button("Submit")
                st.write("En cliquant sur ce bouton, vous acceptez que votre audio soit sauvegardé dans notre base de données.")

                if submit_button:
                    # Ajouter le feedback au DataFrame
                    new_entry = pd.DataFrame([{"filepath": audio_file.name, "prediction": max_emo, "feedback": feedback}])
                    # df = df.append(new_entry, ignore_index=True)
                    df = pd.concat([df, new_entry], ignore_index=True)
                    
                    # Sauvegarder les données mises à jour dans le fichier CSV
                    df.to_csv(csv_file, index=False)

                    # Sauvegarder le fichier audio
                    with open(os.path.join("src","predictions","data",audio_file.name), "wb") as f:
                        f.write(audio_file.getbuffer())
                    
                    # Confirmation pour l'utilisateur
                    st.success("Merci pour votre retour ! Vos données ont été sauvegardées.")