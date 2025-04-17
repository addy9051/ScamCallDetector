import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import threading
import os
from hindifeatures import HindiFeatureExtractor

class AudioProcessor:
    def __init__(self):
        """
        Initialize the audio processor with parameters optimized for Hindi speech.
        
        Based on research from the TeleAntiFraud-28k and HAV-DF datasets, 
        this processor is configured for optimal detection of Hindi scam calls.
        """
        # Audio configuration
        self.sample_rate = 16000  # Standard sample rate for speech (16kHz)
        self.recording = False
        self.audio_data = []
        self.recording_thread = None
        
        # Hindi-specific feature extraction
        self.hindi_extractor = HindiFeatureExtractor()
        
        # Feature extraction configuration
        self.n_mfcc = 20  # Number of MFCC features (increased for better resolution)
        self.hop_length = 512  # Hop length for feature extraction
        self.n_fft = 2048  # FFT window size
        
        # Pre-emphasis filter coefficient (emphasizes higher frequencies)
        # Helps with speech clarity, especially for Hindi consonants
        self.preemphasis = 0.97
        
        print("Audio processor initialized for Hindi speech analysis")
        
    def start_recording(self, output_path):
        """Start recording audio from the microphone."""
        if self.recording:
            print("Already recording")
            return
            
        self.recording = True
        self.audio_data = []
        self.output_path = output_path
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True  # Thread will exit when main program exits
        self.recording_thread.start()
        print("Recording started")
    
    def _record_audio(self):
        """Record audio in a separate thread."""
        try:
            with sd.InputStream(samplerate=self.sample_rate, 
                                channels=1, 
                                callback=self._audio_callback,
                                blocksize=1024,  # Lower block size for better responsiveness
                                dtype='float32'):
                while self.recording:
                    time.sleep(0.1)
            
            # Save recorded audio to file
            if len(self.audio_data) > 0:
                audio_array = np.concatenate(self.audio_data, axis=0)
                
                # Apply pre-emphasis filter to enhance speech clarity
                audio_array = self._apply_preemphasis(audio_array)
                
                # Normalize audio (important for consistent feature extraction)
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array)) * 0.9
                
                # Write to file
                sf.write(self.output_path, audio_array, self.sample_rate)
                print(f"Audio saved to {self.output_path}")
            else:
                print("No audio data recorded")
        except Exception as e:
            print(f"Error in recording audio: {e}")
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio recording."""
        if status:
            print(f"Audio recording error: {status}")
        if self.recording:
            self.audio_data.append(indata.copy())
    
    def _apply_preemphasis(self, signal):
        """
        Apply pre-emphasis filter to the audio signal.
        This emphasizes higher frequencies and improves feature extraction,
        especially for Hindi phonemes.
        """
        return np.append(signal[0], signal[1:] - self.preemphasis * signal[:-1])
    
    def stop_recording(self):
        """Stop the audio recording."""
        if not self.recording:
            print("Not currently recording")
            return
            
        print("Stopping recording...")
        self.recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
            
        print("Recording stopped")
    
    def load_audio(self, file_path):
        """
        Load an audio file and return the waveform and sample rate.
        
        Applies preprocessing steps optimized for Hindi speech analysis:
        1. Resampling to standard rate
        2. Converting to mono
        3. Applying pre-emphasis filter
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return np.array([]), self.sample_rate
            
        try:
            # Load audio file
            waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # Apply pre-emphasis filter to enhance speech clarity
            waveform = self._apply_preemphasis(waveform)
            
            # Trim silence from beginning and end
            waveform, _ = librosa.effects.trim(waveform, top_db=20)
            
            return waveform, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return np.array([]), self.sample_rate
    
    def get_mfccs(self, file_path, n_mfcc=None):
        """
        Extract MFCCs from an audio file with optimizations for Hindi speech.
        
        Args:
            file_path: Path to the audio file
            n_mfcc: Number of MFCC coefficients to extract (default: self.n_mfcc)
            
        Returns:
            numpy.ndarray: MFCC features
        """
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
            
        waveform, sr = self.load_audio(file_path)
        
        if len(waveform) == 0:
            return np.zeros((n_mfcc, 1))
        
        # Extract MFCCs with Hindi-optimized parameters:
        # - Higher number of mel bands for better frequency resolution
        # - Lower frequency range focused on speech (especially Hindi phonemes)
        mfccs = librosa.feature.mfcc(
            y=waveform, 
            sr=sr, 
            n_mfcc=n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=50,  # Lower frequency bound
            fmax=7500,  # Upper frequency bound
            n_mels=40  # Number of Mel bands
        )
        
        # Apply liftering to smooth the MFCCs
        mfccs = librosa.util.normalize(mfccs, axis=1)
        
        return mfccs
    
    def extract_features(self, file_path):
        """
        Extract comprehensive audio features for Hindi scam detection.
        
        Based on TeleAntiFraud-28k and HAV-DF datasets research, this includes:
        - MFCCs (Mel-frequency cepstral coefficients)
        - Spectral contrast features
        - Chroma features
        - Hindi-specific phonetic features
        - Voice quality measurements
        - Scam speech pattern detection
        
        Returns a feature vector ready for model input.
        """
        # Load and preprocess audio
        waveform, sr = self.load_audio(file_path)
        
        if len(waveform) == 0:
            print("Warning: Empty audio file or loading error")
            return np.zeros(100)  # Return zero vector of reasonable size
        
        # 1. Extract standard acoustic features
        
        # MFCCs and derivatives
        mfccs = librosa.feature.mfcc(
            y=waveform, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Spectral contrast (captures voice characteristics)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=waveform, 
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Chroma features (tonal content)
        chroma = librosa.feature.chroma_stft(
            y=waveform, 
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform, 
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform, 
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=waveform, 
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # 2. Get Hindi-specific features using our specialized extractor
        hindi_features = self.hindi_extractor.extract_hindi_features(waveform, sr)
        
        # 3. Advanced phonetic analysis using MFCCs
        phonetic_features = self.hindi_extractor.get_phonetic_features(mfccs)
        
        # 4. Compute voice quality measures (important for detecting synthetic voices)
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(waveform)
        
        # Harmonic-percussive source separation for voice quality analysis
        harmonic, percussive = librosa.effects.hpss(waveform)
        harmonic_ratio = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-8)
        
        # RMS energy (volume patterns)
        rms = librosa.feature.rms(y=waveform)
        
        # 5. Aggregate all features and compute statistics
        feature_list = [
            mfccs, mfcc_delta, mfcc_delta2, 
            spectral_contrast, chroma,
            spectral_centroid, spectral_bandwidth, spectral_rolloff,
            zcr, rms
        ]
        
        # Compute statistical summaries for time-varying features
        combined_features = []
        for feature in feature_list:
            if feature is not None and len(feature) > 0:
                # Compute various statistics to capture the distribution
                combined_features.extend([
                    np.mean(feature, axis=1),     # Central tendency
                    np.std(feature, axis=1),      # Variation
                    np.median(feature, axis=1),   # Robust central tendency
                    np.max(feature, axis=1),      # Maximum values
                    np.min(feature, axis=1),      # Minimum values
                    np.percentile(feature, 25, axis=1),  # Lower quartile
                    np.percentile(feature, 75, axis=1)   # Upper quartile
                ])
        
        # Add harmonic ratio as a scalar feature
        combined_features.append(np.array([harmonic_ratio]))
        
        # 6. Flatten and concatenate all features
        # First collect all the features that can be flattened
        flattened_features = [f.flatten() for f in combined_features if hasattr(f, 'flatten') and f.size > 0]
        
        # Add hindi_features and phonetic_features
        if hindi_features is not None and hindi_features.size > 0:
            flattened_features.append(hindi_features)
        
        if phonetic_features is not None and phonetic_features.size > 0:
            flattened_features.append(phonetic_features)
        
        # Concatenate all features into a single vector
        if flattened_features:
            final_features = np.concatenate([f for f in flattened_features if f.size > 0])
        else:
            print("Warning: No valid features extracted")
            final_features = np.zeros(100)
        
        return final_features
    
    def get_text_from_audio(self, file_path):
        """
        Convert Hindi speech to text using ASR.
        
        In a real implementation, this would use a Hindi-specific ASR model
        like Google's Speech-to-Text API or Hugging Face's Wav2Vec2 model
        fine-tuned for Hindi.
        
        This function now checks if the audio contains actual speech and
        returns appropriate feedback.
        """
        try:
            # Load the audio file
            waveform, sr = self.load_audio(file_path)
            
            # Check if audio contains meaningful sound (not just silence)
            # Calculate RMS energy as a measure of audio volume
            rms_energy = np.sqrt(np.mean(waveform**2))
            
            # Define a threshold for silence
            silence_threshold = 0.01
            
            # If the audio is mostly silent
            if rms_energy < silence_threshold:
                return "No audio detected"
                
            # Get audio duration in seconds
            duration = len(waveform) / sr
            
            # If audio is too short (less than 0.5 seconds), it's likely not speech
            if duration < 0.5:
                return "Audio too short, no speech detected"
            
            # In a real implementation, we would use an actual Hindi ASR model here.
            # For now, we'll indicate that this is a placeholder and actual transcription
            # would require integration with a proper Hindi speech recognition API.
            
            return "Hindi speech detected (transcription unavailable - would require OpenAI Whisper API or other Hindi ASR model)"
            
        except Exception as e:
            print(f"Error in speech detection: {e}")
            return "Error processing audio"
