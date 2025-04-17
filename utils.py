import os
import tempfile
import numpy as np
import re
import string

def get_model_path():
    """
    Return the path to the model directory.
    
    In a real implementation, this would point to a location where models are stored.
    For demonstration, we use a demo_model directory.
    """
    # Create demo_model directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_model")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def format_prediction(prediction, confidence):
    """
    Format the prediction and confidence for display.
    
    Args:
        prediction: Boolean prediction (True for scam, False for not scam)
        confidence: Confidence score (0-100)
        
    Returns:
        tuple: (prediction_text, confidence_text, color)
    """
    if prediction:
        if confidence >= 85:
            prediction_text = "⚠️ HIGH RISK: Scam Detected!"
            confidence_text = f"{confidence:.1f}% confidence - Strong indicators of fraud"
        elif confidence >= 70:
            prediction_text = "⚠️ MEDIUM RISK: Potential Scam"
            confidence_text = f"{confidence:.1f}% confidence - Several suspicious patterns detected"
        else:
            prediction_text = "⚠️ LOW RISK: Possibly Suspicious"
            confidence_text = f"{confidence:.1f}% confidence - Some suspicious elements detected"
        color = "red"
    else:
        if confidence >= 85:
            prediction_text = "✅ SAFE: Legitimate Call"
            confidence_text = f"{confidence:.1f}% confidence - No suspicious patterns detected"
        elif confidence >= 70:
            prediction_text = "✅ LIKELY SAFE: Probably Legitimate"
            confidence_text = f"{confidence:.1f}% confidence - Few suspicious indicators"
        else:
            prediction_text = "⚠️ UNCERTAIN: Exercise Caution"
            confidence_text = f"{confidence:.1f}% confidence - Unable to make confident determination"
        color = "green" if confidence >= 70 else "orange"
    
    return prediction_text, confidence_text, color

def preprocess_text(text):
    """
    Preprocess Hindi text for model input.
    
    This implements Hindi-specific text preprocessing:
    1. Normalize Unicode characters
    2. Remove punctuation while preserving Hindi characters
    3. Remove extra whitespace
    4. Basic cleaning
    
    Args:
        text: Raw Hindi text
        
    Returns:
        str: Processed text ready for model input
    """
    if text is None or not text:
        return ""
    
    # Convert to string if necessary
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Unicode normalization (important for Hindi text)
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Remove Hindi punctuation but keep Hindi characters
    # Hindi Unicode range: \u0900-\u097F
    # Keep Hindi characters and spaces
    hindi_pattern = re.compile(r'[^\u0900-\u097F\s]')
    text = hindi_pattern.sub(' ', text)
    
    # 3. Remove extra whitespace
    text = ' '.join(text.split())
    
    # 4. Basic cleaning
    text = text.strip()
    
    return text

def highlight_scam_keywords(text, scam_keywords):
    """
    Highlight potential scam keywords in the text for display.
    
    Args:
        text: Hindi text to analyze
        scam_keywords: Dictionary of scam keywords with weights
        
    Returns:
        tuple: (highlighted_text, found_keywords)
    """
    if not text or not scam_keywords:
        return text, []
    
    found_keywords = []
    highlighted_text = text
    
    # Find all keywords present in the text
    for keyword in scam_keywords:
        if keyword in text:
            found_keywords.append(keyword)
            # Replace keyword with a highlighted version
            # In a real app, this would use HTML highlighting
            highlighted_text = highlighted_text.replace(
                keyword, f"**{keyword}**"
            )
    
    return highlighted_text, found_keywords

def create_temp_audio_file():
    """Create a temporary file for audio storage."""
    return tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name

def chunk_audio(waveform, sample_rate, chunk_duration=5):
    """
    Split long audio into chunks for better processing.
    
    Long audio files are split into smaller chunks to:
    1. Improve performance with limited memory
    2. Enable more granular analysis (detect specific moments)
    3. Facilitate parallel processing
    
    Args:
        waveform: Audio waveform array
        sample_rate: Sample rate of the audio
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        list: List of audio chunks
    """
    if len(waveform) == 0:
        return []
        
    chunk_size = int(sample_rate * chunk_duration)
    chunks = []
    
    # Calculate 25% overlap between chunks for better detection of events at boundaries
    overlap = int(chunk_size * 0.25)
    hop_size = chunk_size - overlap
    
    # Split into chunks with overlap
    for i in range(0, len(waveform) - overlap, hop_size):
        end_idx = min(i + chunk_size, len(waveform))
        chunk = waveform[i:end_idx]
        
        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
        # Apply a window function to avoid edge effects (Hann window)
        if len(chunk) == chunk_size:
            window = np.hanning(chunk_size)
            chunk = chunk * window
            
        chunks.append(chunk)
    
    return chunks

def extract_acoustic_features(waveform, sr):
    """
    Extract basic acoustic features for display and visualization.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        
    Returns:
        dict: Dictionary of acoustic features
    """
    if len(waveform) == 0:
        return {
            "duration": 0,
            "energy": 0,
            "zero_crossings": 0,
            "silence_ratio": 0
        }
    
    import librosa
    
    # Basic features
    duration = len(waveform) / sr
    energy = np.sum(waveform**2) / len(waveform)
    zero_crossings = np.sum(librosa.zero_crossings(waveform)) / len(waveform)
    
    # Silence detection
    intervals = librosa.effects.split(waveform, top_db=30)
    total_duration = len(waveform) / sr
    speech_duration = sum(i[1] - i[0] for i in intervals) / sr
    silence_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0
    
    return {
        "duration": duration,
        "energy": energy,
        "zero_crossings": zero_crossings,
        "silence_ratio": silence_ratio
    }
