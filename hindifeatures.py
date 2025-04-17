import numpy as np
import librosa

class HindiFeatureExtractor:
    """
    Specialized feature extractor for Hindi speech audio.
    
    This class implements Hindi-specific audio feature extraction methods
    that focus on the unique phonetic characteristics of Hindi speech.
    Based on research from TeleAntiFraud and HAV-DF datasets for Hindi.
    """
    
    def __init__(self):
        """Initialize the Hindi feature extractor with appropriate parameters."""
        # Hindi phoneme frequency ranges based on linguistic research
        # These values are approximated from phonetic studies of Hindi
        self.hindi_phoneme_ranges = {
            # Vowels (svar)
            'a': (600, 900),    # अ
            'aa': (750, 1100),  # आ
            'i': (300, 500),    # इ
            'ii': (350, 600),   # ई
            'u': (350, 500),    # उ
            'uu': (400, 700),   # ऊ
            'e': (400, 600),    # ए
            'ai': (550, 750),   # ऐ
            'o': (450, 650),    # ओ
            'au': (600, 800),   # औ
            # Consonants (vyanjan) frequency ranges by categories
            'velar': (1200, 2500),      # क, ख, ग, घ
            'palatal': (1800, 3000),    # च, छ, ज, झ
            'retroflex': (1600, 2800),  # ट, ठ, ड, ढ
            'dental': (1400, 2600),     # त, थ, द, ध
            'labial': (800, 2000),      # प, फ, ब, भ
            'nasal': (200, 400),        # म, न, ण
            'semi_vowel': (400, 800),   # य, र, ल, व
            'sibilant': (3500, 7000),   # श, ष, स
            'aspirated': (2000, 4000)   # ह and aspirated consonants
        }
        
        # Prosodic features specific to Hindi
        self.hindi_intonation_patterns = {
            'statement': (100, 150),     # Declarative sentence pitch range (Hz)
            'question': (150, 250),      # Interrogative sentence pitch range (Hz)
            'command': (120, 200),       # Imperative sentence pitch range (Hz)
            'scam_urgency': (180, 280)   # Urgency pattern often seen in scam calls
        }
        
        # Common scam speech patterns in Hindi (based on TeleAntiFraud research)
        self.scam_speech_patterns = {
            'rapid_speech': 160,  # Words per minute threshold
            'high_pitch': 200,    # High pitch threshold (Hz)
            'pitch_variation': 50, # Pitch standard deviation threshold
            'pause_ratio': 0.15   # Ratio of silence to speech (lower in scams)
        }
    
    def extract_hindi_features(self, waveform, sample_rate):
        """
        Extract Hindi-specific features from audio based on TeleAntiFraud-28k
        and HAV-DF research.
        
        Args:
            waveform: Audio waveform array
            sample_rate: Sample rate of the audio
            
        Returns:
            numpy.ndarray: Array of Hindi-specific features
        """
        # 1. Pitch and Intonation Analysis (critical for Hindi speech patterns)
        # Extract pitch using more accurate pyin algorithm
        pitches, voiced_flags, voiced_probs = librosa.pyin(
            waveform, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        # Filter valid pitches
        valid_pitches = pitches[voiced_flags]
        
        # Calculate pitch statistics if we have valid pitches
        if len(valid_pitches) > 0:
            pitch_mean = np.mean(valid_pitches)
            pitch_std = np.std(valid_pitches)
            pitch_range = np.max(valid_pitches) - np.min(valid_pitches)
            pitch_change_rate = np.mean(np.abs(np.diff(valid_pitches)))
        else:
            pitch_mean, pitch_std, pitch_range, pitch_change_rate = 0, 0, 0, 0
        
        # 2. Energy Distribution in Hindi Phoneme Ranges
        phoneme_features = []
        spec = np.abs(librosa.stft(waveform))
        freqs = librosa.fft_frequencies(sr=sample_rate)
        
        for phoneme, (low_freq, high_freq) in self.hindi_phoneme_ranges.items():
            # Get the corresponding frequency bins
            low_idx = np.where(freqs >= low_freq)[0][0] if np.any(freqs >= low_freq) else 0
            high_idx = np.where(freqs >= high_freq)[0][0] if np.any(freqs >= high_freq) else len(freqs)-1
            
            # Calculate mean energy in this frequency range
            if low_idx < high_idx and high_idx < spec.shape[0]:
                phoneme_energy = np.mean(spec[low_idx:high_idx, :])
                phoneme_features.append(phoneme_energy)
            else:
                phoneme_features.append(0)
        
        # 3. Rhythm and Tempo Analysis (Hindi has specific rhythmic patterns)
        # Enhanced onset detection with backtracking
        onset_env = librosa.onset.onset_strength(
            y=waveform, 
            sr=sample_rate,
            hop_length=512,
            aggregate=np.median
        )
        # Get tempo and beat frames
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=sample_rate
        )
        # Calculate rhythmic features
        beat_intervals = np.diff(beats) if len(beats) > 1 else np.array([0])
        beat_regularity = np.std(beat_intervals) / np.mean(beat_intervals) if np.mean(beat_intervals) > 0 else 0
        
        # 4. Speech Rate Estimation (fast speech is common in scams)
        # Simplified speech rate estimation
        if len(onset_env) > 0:
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env, 
                sr=sample_rate,
                units='time'
            )
            if len(onsets) > 1:
                speech_rate = len(onsets) / (waveform.shape[0] / sample_rate) * 60  # approx. syllables per minute
            else:
                speech_rate = 0
        else:
            speech_rate = 0
        
        # 5. Voice Quality Measures (for detecting synthesized or stressed voice)
        # Spectral centroid (brightness of sound)
        centroid = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sample_rate))
        # Spectral flatness (for tonal vs. noise distinction)
        flatness = np.mean(librosa.feature.spectral_flatness(y=waveform))
        # Zero crossing rate (related to voice tension)
        zcr = np.mean(librosa.feature.zero_crossing_rate(waveform))
        
        # 6. Pause Analysis (scammers often use fewer pauses)
        # Detect silence
        intervals = librosa.effects.split(waveform, top_db=30)
        if len(waveform) > 0:
            speech_duration = sum(i[1] - i[0] for i in intervals) / sample_rate
            total_duration = len(waveform) / sample_rate
            pause_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0
        else:
            pause_ratio = 0
        
        # 7. Formant Analysis (important for vowel distinction in Hindi)
        # Using a simple approximation of formant tracking
        formants = []
        for i in range(1, 5):  # F1-F4 formant frequencies
            formant_range = (i * 500, i * 1000)  # Approximate formant ranges
            low_idx = np.where(freqs >= formant_range[0])[0][0] if np.any(freqs >= formant_range[0]) else 0
            high_idx = np.where(freqs >= formant_range[1])[0][0] if np.any(freqs >= formant_range[1]) else len(freqs)-1
            
            if low_idx < high_idx and high_idx < spec.shape[0]:
                # Find peaks in this range (simplified formant detection)
                formant_slice = np.mean(spec[low_idx:high_idx, :], axis=1)
                if len(formant_slice) > 0:
                    peak_idx = np.argmax(formant_slice)
                    formant_freq = freqs[low_idx + peak_idx]
                    formant_energy = formant_slice[peak_idx]
                    formants.extend([formant_freq, formant_energy])
                else:
                    formants.extend([0, 0])
            else:
                formants.extend([0, 0])
        
        # 8. Check for speech patterns common in scams
        scam_pattern_features = [
            # is speech rate above the scam threshold?
            1.0 if speech_rate > self.scam_speech_patterns['rapid_speech'] else 0.0,
            # is average pitch high (like in urgent messages)?
            1.0 if pitch_mean > self.scam_speech_patterns['high_pitch'] else 0.0,
            # is pitch variation high (emotional manipulation)?
            1.0 if pitch_std > self.scam_speech_patterns['pitch_variation'] else 0.0,
            # are there fewer pauses than normal conversation?
            1.0 if pause_ratio < self.scam_speech_patterns['pause_ratio'] else 0.0
        ]
        
        # 9. Harmonic-to-noise ratio (voice quality measure)
        harmonic_ratio = np.mean(librosa.effects.harmonic(waveform)) / np.mean(waveform) if np.mean(waveform) != 0 else 0
        
        # Combine all Hindi-specific features
        hindi_features = np.array([
            # Pitch features
            pitch_mean, pitch_std, pitch_range, pitch_change_rate,
            # Rhythm features
            tempo, beat_regularity,
            # Speech rate
            speech_rate,
            # Voice quality
            centroid, flatness, zcr, harmonic_ratio,
            # Pause analysis
            pause_ratio,
            # Add all phoneme energy features
            *phoneme_features,
            # Add formant analysis
            *formants,
            # Add scam pattern features
            *scam_pattern_features
        ])
        
        return hindi_features
    
    def get_phonetic_features(self, mfccs):
        """
        Extract phonetic features from MFCCs that are particularly 
        relevant for Hindi speech patterns and scam detection.
        
        Args:
            mfccs: MFCC features
            
        Returns:
            numpy.ndarray: Phonetic features
        """
        if mfccs.size == 0:
            return np.array([])
        
        # 1. Calculate delta and delta-delta coefficients
        # These capture the dynamics of speech articulation
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # 2. Extract statistics relevant to Hindi phonetic patterns
        phonetic_features = []
        
        # Standard statistical features for each coefficient set
        for feat, name in zip([mfccs, mfcc_delta, mfcc_delta2], 
                             ["mfcc", "delta", "delta2"]):
            # Central moments
            phonetic_features.extend([
                np.mean(feat, axis=1),  # Average
                np.std(feat, axis=1),   # Variation
                np.median(feat, axis=1),  # Median (robust to outliers)
                np.max(feat, axis=1),   # Maximum 
                np.min(feat, axis=1),   # Minimum
                np.percentile(feat, 25, axis=1),  # 1st quartile
                np.percentile(feat, 75, axis=1),  # 3rd quartile
            ])
            
            # Shape features
            if feat.shape[1] > 1:
                phonetic_features.extend([
                    np.mean(np.diff(feat, axis=1), axis=1),  # Average rate of change
                    np.std(np.diff(feat, axis=1), axis=1),   # Variability in rate of change
                ])
        
        # 3. Extract Hindi-specific coefficient patterns
        # Focus on certain coefficients more relevant to Hindi phonemes
        # Lower coefficients (2-5) capture vowel qualities important in Hindi
        vowel_coeffs = mfccs[1:5, :] if mfccs.shape[0] > 5 else mfccs
        # Mid coefficients (5-9) relate to consonant distinctions
        consonant_coeffs = mfccs[5:9, :] if mfccs.shape[0] > 9 else mfccs
        
        # Add specialized features for these ranges
        for coeffs, name in zip([vowel_coeffs, consonant_coeffs], 
                               ["vowel", "consonant"]):
            if coeffs.size > 0:
                phonetic_features.extend([
                    np.mean(coeffs, axis=1),
                    np.std(coeffs, axis=1)
                ])
        
        # 4. Temporal pattern features
        # Rhythm patterns are important in Hindi prosody
        if mfccs.shape[1] > 2:
            # Auto-correlation to capture rhythmic patterns
            autocorr = np.array([
                np.correlate(mfccs[i, :], mfccs[i, :], mode='full')[mfccs.shape[1]-1:] 
                for i in range(min(5, mfccs.shape[0]))
            ])
            if autocorr.size > 0:
                # Extract peak positions and values of autocorrelation
                peak_features = []
                for i in range(min(autocorr.shape[0], 5)):
                    if autocorr.shape[1] > 3:
                        peaks = np.argsort(autocorr[i, 1:min(10, autocorr.shape[1])])[-3:]
                        peak_vals = autocorr[i, peaks+1]
                        peak_features.extend(peaks)
                        peak_features.extend(peak_vals)
                    
                phonetic_features.extend(peak_features)
        
        # Flatten all features into a single vector
        flat_features = [f.flatten() for f in phonetic_features if hasattr(f, 'flatten') and f.size > 0]
        scalar_features = [f for f in phonetic_features if not hasattr(f, 'flatten') and np.isscalar(f)]
        
        # Combine all features
        if flat_features and scalar_features:
            return np.concatenate(flat_features + [np.array(scalar_features)])
        elif flat_features:
            return np.concatenate(flat_features)
        elif scalar_features:
            return np.array(scalar_features)
        else:
            return np.array([])
