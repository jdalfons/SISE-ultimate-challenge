import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import io
import json
from model.transcriber import transcribe_audio
from predict import predict_emotion

# You'll need to install this package:
# pip install streamlit-audiorec
from st_audiorec import st_audiorec

AUDIO_WAV = 'audio/wav'
MAX_FILE_SIZE_MB = 10
# Page configuration
st.set_page_config(
    page_title="Emotional Report Analyzer",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = []
if 'current_audio_index' not in st.session_state:
    st.session_state.current_audio_index = -1
if 'audio_history_csv' not in st.session_state:
    # Define columns for our CSV storage
    st.session_state.audio_history_csv = pd.DataFrame(
        columns=['timestamp', 'file_path', 'transcription', 'emotion', 'probabilities']
    )
if 'needs_rerun' not in st.session_state:
    st.session_state.needs_rerun = False

# Function to ensure we keep only the last 10 entries
def update_audio_history(new_entry):
    # Add the new entry
    st.session_state.audio_history_csv = pd.concat([st.session_state.audio_history_csv, pd.DataFrame([new_entry])], ignore_index=True)
    
    # Keep only the last 10 entries
    if len(st.session_state.audio_history_csv) > 10:
        st.session_state.audio_history_csv = st.session_state.audio_history_csv.iloc[-10:]
    
    # Save to CSV
    st.session_state.audio_history_csv.to_csv('audio_history.csv', index=False)

# Function to process audio and get results
def process_audio(audio_path):
    try:
        # Get transcription
        transcription = transcribe_audio(audio_path)
        
        # Get emotion prediction
        predicted_emotion, probabilities = predict_emotion(audio_path)
        
        # Update audio history
        new_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'file_path': audio_path,
            'transcription': transcription,
            'emotion': predicted_emotion,
            'probabilities': str(probabilities)  # Convert dict to string for storage
        }
        update_audio_history(new_entry)
        
        # Update current index
        st.session_state.current_audio_index = len(st.session_state.audio_history_csv) - 1
        
        return transcription, predicted_emotion, probabilities
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None, None

# Function to split audio into 10-second segments
def split_audio(audio_file, segment_length=10):
    # This is a placeholder - in a real implementation, you'd use a library like pydub
    # to split the audio file into segments
    st.warning("Audio splitting functionality is a placeholder. Implement with pydub or similar library.")
    # For now, we'll just return the whole file as a single segment
    return [audio_file]

# Function to display emotion visualization
def display_emotion_chart(probabilities):
    emotions = list(probabilities.keys())
    values = list(probabilities.values())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(emotions, values, color=['red', 'gray', 'green'])
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probability')
    ax.set_title('Emotion Prediction Results')
    
    st.pyplot(fig)

# Trigger rerun if needed (replaces experimental_rerun)
if st.session_state.needs_rerun:
    st.session_state.needs_rerun = False
    st.rerun()  # Using st.rerun() instead of experimental_rerun

col_logo, col_name = st.columns([3, 1])
col_logo.image("./img/logo_01.png", width=400)
col_name.title("Emotional Report")

# Create two columns for the main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Audio Input")
    
    # Method selection

    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])
    
    with tab1:
        st.write("Record your audio (max 10 seconds):")
        
        # Using streamlit-audiorec for better recording functionality
        wav_audio_data = st_audiorec()
        
        if wav_audio_data is not None:
            # Save the recorded audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(wav_audio_data)
                tmp_file_path = tmp_file.name
            
            st.success("Audio recorded successfully!")
            
            # Process button
            if st.button("Process Recorded Audio"):
                # Process the audio
                with st.spinner("Processing audio..."):
                    transcription, emotion, probs = process_audio(tmp_file_path)
                    # Set flag for rerun instead of calling experimental_rerun
                    if transcription is not None:
                        st.success("Audio processed successfully!")
                        st.session_state.needs_rerun = True
    
    with tab2:
        uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=['wav'])
        
        if uploaded_file is not None and uploaded_file.type == AUDIO_WAV and uploaded_file.size < MAX_FILE_SIZE_MB * 1_000_000:
            try:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name
            except Exception as e:
                st.error(f"Error saving uploaded file: {str(e)}")
                st.error(f"Try to record your voice directly, maybe your storage is locked.")
                
            st.audio(uploaded_file, format="audio/wav")
            
            # Process button
            if st.button("Process Uploaded Audio"):
                # Split audio into 10-second segments
                with st.spinner("Processing audio..."):
                    segments = split_audio(tmp_file_path)
                    
                    # Process each segment
                    for i, segment_path in enumerate(segments):
                        st.write(f"Processing segment {i+1}...")
                        transcription, emotion, probs = process_audio(segment_path)
                        
                    # Set flag for rerun instead of calling experimental_rerun
                    st.success("Audio processed successfully!")
                    st.session_state.needs_rerun = True
    # Audio History and Analytics Section
    st.header("Audio History and Analytics")

    if len(st.session_state.audio_history_csv) > 0:
        # Display a select box to choose from audio history
        timestamps = st.session_state.audio_history_csv['timestamp'].tolist()
        selected_timestamp = st.selectbox(
            "Select audio from history:",
            options=timestamps,
            index=len(timestamps) - 1  # Default to most recent
        )
        
        # Update current index when selection changes
        selected_index = st.session_state.audio_history_csv[
            st.session_state.audio_history_csv['timestamp'] == selected_timestamp
        ].index[0]
        
        # Only update if different
        if st.session_state.current_audio_index != selected_index:
            st.session_state.current_audio_index = selected_index
            st.session_state.needs_rerun = True
        
        # Analytics button
        if st.button("Run Analytics on Selected Audio"):
            st.subheader("Analytics Results")
            
            # Get the selected audio data
            selected_data = st.session_state.audio_history_csv.iloc[selected_index]
            
            # Display analytics (this is where you would add more sophisticated analytics)
            st.write(f"Selected Audio: {selected_data['timestamp']}")
            st.write(f"Emotion: {selected_data['emotion']}")
            st.write(f"File Path: {selected_data['file_path']}")
            
            # Add any additional analytics you want here
            
            # Try to play the selected audio
            try:
                if os.path.exists(selected_data['file_path']):
                    st.audio(selected_data['file_path'], format="audio/wav")
                else:
                    st.warning("Audio file not found - it may have been deleted or moved.")
            except Exception as e:
                st.error(f"Error playing audio: {str(e)}")
    else:
        st.info("No audio history available. Record or upload audio to create history.")
        
with col2:
    st.header("Results")
    
    # Display results if available
    if st.session_state.current_audio_index >= 0 and len(st.session_state.audio_history_csv) > 0:
        current_data = st.session_state.audio_history_csv.iloc[st.session_state.current_audio_index]
        
        # Transcription
        st.subheader("Transcription")
        st.text_area("", value=current_data['transcription'], height=100, key="transcription_area")
        
        # Emotion
        st.subheader("Detected Emotion")
        st.info(f"🎭 Predicted emotion: **{current_data['emotion']}**")
        
        # Convert string representation of dict back to actual dict
        try:
            import ast
            probs = ast.literal_eval(current_data['probabilities'])
            display_emotion_chart(probs)
        except Exception as e:
            st.error(f"Error parsing probabilities: {str(e)}")
            st.write(f"Raw probabilities: {current_data['probabilities']}")
    else:
        st.info("Record or upload audio to see results")



# Footer
st.markdown("---")
st.caption("Emotional Report Analyzer - Processes audio in 10-second segments and predicts emotions")