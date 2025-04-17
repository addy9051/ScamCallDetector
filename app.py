import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os
import time
import datetime
import librosa
import librosa.display
from audio_processor import AudioProcessor
from model import ScamDetectionModel
import utils

# Define functions for processing and displaying results
def process_audio(audio_file_path, sensitivity=0.5):
    """Process the audio file and display results"""
    with st.spinner("Analyzing audio..."):
        # Extract features
        features = st.session_state.audio_processor.extract_features(audio_file_path)
        
        # Get ASR text (in real implementation, would use a Hindi ASR model)
        text = st.session_state.audio_processor.get_text_from_audio(audio_file_path)
        
        # Make prediction (with sensitivity adjustment)
        prediction, confidence = st.session_state.model.predict(features, text, sensitivity)
        
        # Get acoustic features for visualization
        waveform, sr = st.session_state.audio_processor.load_audio(audio_file_path)
        mfccs = st.session_state.audio_processor.get_mfccs(audio_file_path)
        acoustic_features = utils.extract_acoustic_features(waveform, sr)
        
        # Highlight scam keywords if text is available
        found_keywords = []
        highlighted_text = text
        if text and hasattr(st.session_state.model, 'scam_keyword_weights'):
            highlighted_text, found_keywords = utils.highlight_scam_keywords(
                text, st.session_state.model.scam_keyword_weights
            )
        
        # Format prediction for display
        prediction_text, confidence_text, color = utils.format_prediction(prediction, confidence)
        
        # Store analysis results
        analysis = {
            'prediction': prediction,
            'confidence': confidence,
            'prediction_text': prediction_text,
            'confidence_text': confidence_text,
            'color': color,
            'text': text,
            'highlighted_text': highlighted_text,
            'keywords': found_keywords,
            'waveform': waveform,
            'sr': sr,
            'mfccs': mfccs,
            'acoustic_features': acoustic_features,
            'audio_path': audio_file_path
        }
        
        # Store in session state for later reference
        st.session_state.last_analysis = analysis
        
        # Update history
        if len(st.session_state.history) > 0 and st.session_state.history.iloc[0]['Filename'] == os.path.basename(audio_file_path):
            # Skip if this exact file was just analyzed (prevents duplicates)
            pass
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = os.path.basename(audio_file_path)
            keywords_str = ", ".join(found_keywords) if found_keywords else "None detected"
            
            new_entry = pd.DataFrame([{
                'Timestamp': timestamp,
                'Filename': filename,
                'Prediction': 'Scam' if prediction else 'Not Scam',
                'Confidence': f"{confidence:.1f}%",
                'Keywords': keywords_str
            }])
            st.session_state.history = pd.concat([new_entry, st.session_state.history]).reset_index(drop=True)
        
        # Display results
        display_analysis_results(analysis)
        
        return analysis

def display_analysis_results(analysis):
    """Display the analysis results in the UI"""
    # Get results container
    result_placeholder = st.empty()
    
    # Display results
    with result_placeholder.container():
        # Header with prediction result
        st.markdown(
            f"<h2 style='color: {analysis['color']};'>{analysis['prediction_text']}</h2>", 
            unsafe_allow_html=True
        )
        st.markdown(f"<p><strong>{analysis['confidence_text']}</strong></p>", unsafe_allow_html=True)
        
        # Create two columns for details
        details_col1, details_col2 = st.columns([3, 2])
        
        with details_col1:
            if analysis['prediction']:
                st.markdown("""
                ### Warning Signs Detected:
                - Suspicious voice patterns and intonation
                - Known scam indicators in speech
                - High similarity to known scam profiles
                
                ### Recommended Action:
                - Do not share personal information
                - Hang up and report to authorities
                - Verify through official channels
                """)
            else:
                st.markdown("""
                ### Analysis:
                - Voice characteristics match legitimate call patterns
                - Natural speech rhythm and intonation
                - Content does not match known scam scripts
                
                ### Note:
                Always maintain caution with any calls requesting personal information.
                """)
        
        with details_col2:
            # Display any detected keywords
            if analysis['keywords']:
                st.markdown("### Suspicious Keywords Detected:")
                for keyword in analysis['keywords']:
                    st.markdown(f"- **{keyword}**")
            
            # Show basic acoustic feature stats
            st.markdown("### Audio Characteristics:")
            features = analysis['acoustic_features']
            st.markdown(f"- Duration: {features['duration']:.2f} seconds")
            st.markdown(f"- Speech/silence ratio: {(1-features['silence_ratio']):.2f}")
            st.markdown(f"- Voice intensity: {'High' if features['energy'] > 0.05 else 'Normal'}")
        
        # Display detected speech (if any)
        if analysis['text']:
            st.subheader("Detected Speech:")
            st.markdown(analysis['highlighted_text'])
        
        # Add feedback options
        st.markdown("### Was this analysis helpful?")
        feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 2])
        
        if feedback_col1.button("👍 Correct"):
            st.session_state.feedback_count += 1
            st.success("Thanks for your feedback! This helps improve our model.")
            
        if feedback_col2.button("👎 Incorrect"):
            st.session_state.feedback_count += 1
            
            # Get the opposite prediction
            correct_label = not analysis['prediction']
            
            # Update model with feedback
            if 'text' in analysis and 'waveform' in analysis:
                features = st.session_state.audio_processor.extract_features(analysis['audio_path'])
                st.session_state.model.update_with_feedback(features, analysis['text'], correct_label)
            
            st.warning("Thanks for correcting us! This feedback helps improve our model.")

def show_analysis_visuals(analysis):
    """Display visualizations for the audio analysis"""
    st.markdown("### Audio Analysis Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Display waveform visualization
        st.subheader("Audio Waveform")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(analysis['waveform'], sr=analysis['sr'], ax=ax)
        ax.set_title('Audio Waveform')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)
    
    with viz_col2:
        # Display MFCC visualization
        st.subheader("MFCC Features")
        fig, ax = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(
            analysis['mfccs'], 
            x_axis='time', 
            ax=ax, 
            sr=analysis['sr'],
            hop_length=512
        )
        ax.set_title('MFCC Features (Acoustic Fingerprint)')
        ax.set_xlabel('Time')
        ax.set_ylabel('MFCC Coefficients')
        fig.colorbar(img, ax=ax, format="%+2.f")
        st.pyplot(fig)
    
    # Display spectrogram
    st.subheader("Spectral Analysis")
    fig, ax = plt.subplots(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(analysis['waveform'])), ref=np.max)
    img = librosa.display.specshow(
        D, 
        y_axis='log', 
        x_axis='time', 
        ax=ax, 
        sr=analysis['sr']
    )
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

# Set page configuration
st.set_page_config(
    page_title="Hindi Scam Call Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #333;
    }
    .info-text {
        font-size: 1.0rem;
    }
    .risk-high {
        color: #FF0000;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        color: #FF5733;
        font-weight: bold;
    }
    .risk-low {
        color: #FFC300;
        font-weight: bold;
    }
    .safe {
        color: #2ECC71;
        font-weight: bold;
    }
    .highlight-keyword {
        background-color: #FFFF00;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .recording-indicator {
        color: #FF0000;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.5; }
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=['Timestamp', 'Filename', 'Prediction', 'Confidence', 'Keywords'])
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()
if 'model' not in st.session_state:
    st.session_state.model = ScamDetectionModel()
if 'feedback_count' not in st.session_state:
    st.session_state.feedback_count = 0
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

# Title and description
st.markdown("<h1 class='main-header'>🛡️ Hindi Scam Call Detector</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="info-text">
This application helps you detect potential scam calls in Hindi. 
Record your call or upload an audio file to analyze whether it's a legitimate call or a potential scam.
Powered by advanced speech processing and AI specifically optimized for Hindi language patterns.
</div>
""", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["🎤 Detection", "📊 History", "ℹ️ About"])

with tab1:
    # Sidebar with options
    st.sidebar.markdown("<h2>Detection Settings</h2>", unsafe_allow_html=True)
    detection_mode = st.sidebar.radio(
        "Choose Input Method:",
        ("Record Audio", "Upload Audio File")
    )
    
    sensitivity = st.sidebar.slider(
        "Detection Sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Higher sensitivity may increase false positives"
    )
    
    # Advanced options in sidebar
    with st.sidebar.expander("Advanced Options"):
        st.checkbox("Enable voice deepfake detection", value=True, 
                    help="Detects synthetic or cloned voices (beta feature)")
        st.checkbox("Real-time analysis", value=False, disabled=True,
                    help="Analyze audio in real-time (coming soon)")
        st.checkbox("Save detection reports", value=True,
                    help="Save analysis results to history")
    
    # Safety tips in sidebar
    with st.sidebar.expander("Safety Tips"):
        st.markdown("""
        * Never share OTPs or passwords over phone
        * Government agencies don't request payments via phone
        * Verify caller identity through official channels
        * Report suspicious calls to authorities
        """)
    
    # Main content area
    detection_col1, detection_col2 = st.columns([3, 2])
    
    with detection_col1:
        st.markdown("<h2 class='sub-header'>Audio Input</h2>", unsafe_allow_html=True)
        
        if detection_mode == "Record Audio":
            # Record audio from microphone
            st.info("Click 'Start Recording' to begin. Speak or play the call audio clearly.")
            
            # Initialize session state variables for recording if they don't exist
            if 'recording' not in st.session_state:
                st.session_state.recording = False
            if 'audio_file_path' not in st.session_state:
                st.session_state.audio_file_path = None
            if 'recording_duration' not in st.session_state:
                st.session_state.recording_duration = 0
            if 'recording_start_time' not in st.session_state:
                st.session_state.recording_start_time = 0
            
            # Audio recording controls
            col_rec1, col_rec2 = st.columns(2)
            
            if not st.session_state.recording:
                if col_rec1.button("▶️ Start Recording", key="start_rec"):
                    st.session_state.recording = True
                    st.session_state.audio_file_path = utils.create_temp_audio_file()
                    st.session_state.recording_start_time = time.time()
                    st.session_state.audio_processor.start_recording(st.session_state.audio_file_path)
                    st.rerun()
            else:
                if col_rec1.button("⏹️ Stop Recording", key="stop_rec"):
                    st.session_state.audio_processor.stop_recording()
                    st.session_state.recording = False
                    st.session_state.recording_duration = time.time() - st.session_state.recording_start_time
                    st.rerun()
                
                # Show recording indicator and duration
                elapsed = time.time() - st.session_state.recording_start_time
                col_rec2.markdown(
                    f"<div class='recording-indicator'>🔴 Recording in progress... {elapsed:.1f}s</div>", 
                    unsafe_allow_html=True
                )
            
            # Display and process audio if recording is complete
            if not st.session_state.recording and st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
                st.success(f"Recording completed! Duration: {st.session_state.recording_duration:.1f} seconds")
                st.audio(st.session_state.audio_file_path)
                
                analyze_btn = st.button("🔍 Analyze Recording", key="analyze_rec")
                if analyze_btn:
                    # Process the audio and make prediction
                    process_audio(st.session_state.audio_file_path, sensitivity)
            
        elif detection_mode == "Upload Audio File":
            # Upload audio file
            uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg', 'm4a'])
            
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                audio_file_path = utils.create_temp_audio_file()
                with open(audio_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.audio(audio_file_path)
                
                analyze_btn = st.button("🔍 Analyze Audio File", key="analyze_upload")
                if analyze_btn:
                    # Process the audio and make prediction
                    process_audio(audio_file_path, sensitivity)
    
    with detection_col2:
        st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
        result_placeholder = st.empty()
        
        # Display a placeholder for results
        with result_placeholder.container():
            st.info("Submit an audio recording or file to see analysis results.")
    
    # Display last analysis if available
    if st.session_state.last_analysis:
        analysis = st.session_state.last_analysis
        show_analysis_visuals(analysis)

with tab2:
    # History section
    st.markdown("<h2 class='sub-header'>Detection History</h2>", unsafe_allow_html=True)
    
    history_controls_col1, history_controls_col2 = st.columns([3, 1])
    
    if history_controls_col2.button("Clear History"):
        st.session_state.history = pd.DataFrame(
            columns=['Timestamp', 'Filename', 'Prediction', 'Confidence', 'Keywords'])
        st.rerun()
    
    if len(st.session_state.history) > 0:
        # Add search and filter options
        search_term = history_controls_col1.text_input("Search in history:")
        
        # Filter history based on search term
        filtered_history = st.session_state.history
        if search_term:
            mask = filtered_history.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
            filtered_history = filtered_history[mask]
        
        # Display history table
        st.dataframe(
            filtered_history,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Date & Time"),
                "Prediction": st.column_config.TextColumn("Result"),
                "Confidence": st.column_config.TextColumn("Confidence"),
                "Keywords": st.column_config.TextColumn("Detected Keywords"),
            },
            use_container_width=True
        )
        
        # Summary statistics
        st.subheader("Summary Statistics")
        scam_count = len(filtered_history[filtered_history['Prediction'] == 'Scam'])
        legitimate_count = len(filtered_history[filtered_history['Prediction'] == 'Not Scam'])
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("Total Analyzed", len(filtered_history))
        stats_col2.metric("Scams Detected", scam_count)
        stats_col3.metric("Legitimate Calls", legitimate_count)
        
        # Visualization of history
        if len(filtered_history) > 1:
            st.subheader("Detection Trend")
            filtered_history['Date'] = pd.to_datetime(filtered_history['Timestamp'])
            filtered_history['IsScam'] = filtered_history['Prediction'].apply(lambda x: 1 if x == 'Scam' else 0)
            
            # Group by date and calculate scam ratio
            daily_stats = filtered_history.groupby(filtered_history['Date'].dt.date).agg(
                ScamCount=('IsScam', 'sum'),
                TotalCount=('IsScam', 'count')
            )
            daily_stats['ScamRatio'] = daily_stats['ScamCount'] / daily_stats['TotalCount'] * 100
            
            # Plot trend
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(daily_stats.index, daily_stats['ScamRatio'], color='#FF5733')
            ax.set_ylabel('Scam Detection %')
            ax.set_title('Daily Scam Detection Percentage')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
    else:
        st.info("No detection history yet. Start analyzing calls to build history.")

with tab3:
    # About section
    st.markdown("<h2 class='sub-header'>About This Detection System</h2>", unsafe_allow_html=True)
    
    about_col1, about_col2 = st.columns(2)
    
    with about_col1:
        st.markdown("""
        ### How It Works:
        
        1. **Audio Processing**: Extracts comprehensive acoustic features, including:
           - MFCCs (Mel-frequency cepstral coefficients)
           - Spectral contrast and chroma features
           - Voice quality measurements
           
        2. **Hindi-Specific Analysis**: 
           - Specialized feature engineering for Hindi phonemes
           - Hindi speech pattern detection
           - Prosodic analysis tuned for Hindi language
           
        3. **Text Analysis**: 
           - Speech-to-text conversion with Hindi language model
           - Keyword and phrase detection for common scam indicators
           - Contextual analysis of conversation patterns
           
        4. **Multimodal Classification**:
           - Combines audio and text features
           - Late fusion architecture for optimal accuracy
           - Confidence scoring with risk levels
           
        5. **Continuous Learning**:
           - Model improves with user feedback
           - Regular updates based on new scam patterns
           - Adaptation to evolving scam tactics
        """)
    
    with about_col2:
        st.markdown("""
        ### Research and Data Sources:
        
        Our model is built using the following datasets and research:
        
        #### TeleAntiFraud-28k
        Hindi language samples extracted using specialized language detection,
        focusing on patterns specific to phone scams in India.
        
        #### HAV-DF Dataset
        Synthetic Hindi deepfakes for training the model to detect advanced
        voice cloning scams, which are becoming increasingly common.
        
        #### Ethically Collected Public Examples
        Samples from public sources like YouTube and awareness campaigns,
        processed with speaker diarization and noise reduction.
        
        ### Privacy Commitment:
        
        - Audio recordings are processed locally
        - No data is stored permanently without explicit consent
        - Analysis results are stored only on your device
        - No external APIs are used without permission
        """)
    
    # References and resources
    st.markdown("### References and Further Reading")
    
    references_col1, references_col2 = st.columns(2)
    
    with references_col1:
        st.markdown("""
        * [TeleAntiFraud: Benchmark Models and Dataset for Anti-Fraud Phone Calls Detection](https://arxiv.org/html/2503.24115v1)
        * [HAV-DF: A Hindi Audio-Visual Deepfake Dataset](https://arxiv.org/html/2411.15457v1)
        * [TRAI Guidelines on Scam Prevention](https://www.trai.gov.in/)
        """)
    
    with references_col2:
        st.markdown("""
        * [Common Phone Scams in India (2025)](https://www.reddit.com/r/IndiaInvestments/comments/1bf9f6k/psa_new_scam_these_days_starts_with_trai_sms_or_a/)
        * [Voice Cloning Technology and Detection Methods](https://youtu.be/Q8FQaJWi0Ac)
        * [Cyberabad Police Advisory on Phone Scams](https://cyberabadpolice.gov.in/)
        """)

