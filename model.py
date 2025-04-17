import numpy as np
import os
import pickle
import random
import time
from utils import get_model_path

class ScamDetectionModel:
    def __init__(self):
        """
        Initialize the scam detection model.
        
        In production, this would load a real trained model. 
        For demonstration purposes, we're using a placeholder model.
        """
        self.model_loaded = False
        self.model_path = get_model_path()
        
        # Create a placeholder model for demonstration
        print("Using placeholder model for demonstration.")
        self._create_placeholder_model()
    
    def load_model(self):
        """
        Load the trained model from disk.
        
        This would load both the audio and text models, as well as
        the fusion model that combines their outputs.
        
        For now, this is just a placeholder as we're using a rule-based approach.
        """
        pass
    
    def _create_placeholder_model(self):
        """Create a placeholder model for demonstration purposes."""
        # This is a placeholder to simulate model behavior
        # In a real implementation, these would be actual trained models
        self.model_loaded = False
        self.hindi_keywords = {
            "scam": ["धोखा", "फ्रॉड", "तुरंत", "जल्दी", "पैसा", "नकद", "खाता", "पासवर्ड"],
            "safe": ["धन्यवाद", "सहायता", "समर्थन", "जानकारी", "आभार"]
        }
        
        # Add more scam keywords based on common Hindi scam patterns
        self.scam_keyword_weights = {
            "धोखा": 0.7,      # fraud
            "फ्रॉड": 0.8,      # fraud
            "तुरंत": 0.4,      # immediate/urgent
            "जल्दी": 0.4,      # hurry
            "पैसा": 0.3,       # money
            "नकद": 0.5,        # cash
            "खाता": 0.4,       # account
            "पासवर्ड": 0.7,    # password
            "अभी": 0.4,        # now
            "बैंक": 0.3,       # bank
            "कार्ड": 0.5,      # card
            "ओटीपी": 0.7,      # OTP
            "लॉटरी": 0.8,      # lottery
            "इनाम": 0.6,       # prize
            "गोपनीय": 0.5      # confidential
        }
        
        # Safe keywords with negative weights (reduce scam score)
        self.safe_keyword_weights = {
            "धन्यवाद": -0.3,   # thank you
            "सहायता": -0.2,    # help
            "समर्थन": -0.2,    # support
            "जानकारी": -0.1,   # information
            "आभार": -0.3       # gratitude
        }
    
    def predict(self, audio_features, text=None, sensitivity=0.5):
        """
        Make a prediction about whether an audio clip is a scam or not.
        
        Args:
            audio_features: Extracted audio features
            text: Transcribed text (if available)
            sensitivity: Detection sensitivity (0.0-1.0), higher values make the model
                        more likely to classify calls as scams
            
        Returns:
            tuple: (is_scam (bool), confidence (float))
        """
        # Simulate prediction for demonstration
        time.sleep(0.5)  # Simulate processing time (reduced for better UX)
        
        # Base score (adjusted by sensitivity)
        # Lower sensitivity means we start with a lower base score (less likely to be scam)
        # Higher sensitivity means we start with a higher base score (more likely to be scam)
        base_score_adjustment = (sensitivity - 0.5) * 0.2
        scam_score = 0.35 + base_score_adjustment
        
        # Analyze text if available
        if text is not None and text:
            # Analyze text for scam keywords
            for keyword, weight in self.scam_keyword_weights.items():
                if keyword in text:
                    # Adjust keyword weight based on sensitivity
                    adjusted_weight = weight * (0.8 + sensitivity * 0.4)
                    scam_score += adjusted_weight * 0.1
            
            # Check for safe keywords that might reduce the scam score
            for keyword, weight in self.safe_keyword_weights.items():
                if keyword in text:
                    # For safe keywords, lower sensitivity reduces their impact
                    # (making the model more cautious)
                    discount_factor = 1.0 - (sensitivity * 0.4)
                    scam_score += weight * 0.1 * discount_factor
        
        # Analyze audio features
        # In a real model, we would analyze specific patterns in the audio features
        # For this demo, we'll use some heuristics
        
        # 1. Check for rapid speech (a common scam indicator)
        if len(audio_features) > 10:
            feature_avg = np.mean(audio_features[:10])
            # Scale this to a small contribution to the score, influenced by sensitivity
            scam_score += (feature_avg - np.mean(audio_features)) * 0.01 * (1.0 + sensitivity)
        
        # 2. Check the variance of features as a proxy for speech variability
        # Scammers often have more variable speech patterns
        feature_variance = np.var(audio_features)
        variance_threshold = 1.0 - (sensitivity * 0.5)  # Lower threshold with higher sensitivity
        if feature_variance > variance_threshold:
            scam_score += 0.05 * (1.0 + sensitivity)
        
        # 3. Check for extreme values in audio features
        # Scammers may have more extreme emotional patterns
        if len(audio_features) > 0:
            max_feature = np.max(np.abs(audio_features))
            if max_feature > 5.0:  # Arbitrary threshold for extreme values
                scam_score += 0.05 * sensitivity
        
        # Add some randomness for demonstration purposes
        # The randomness is reduced with higher sensitivity for more consistent results
        randomness_factor = 0.2 * (1.0 - sensitivity * 0.5)
        scam_score += (random.random() - 0.5) * randomness_factor
        
        # Adjust threshold based on sensitivity
        # Higher sensitivity means lower threshold to classify as scam
        threshold = 0.5 - (sensitivity - 0.5) * 0.2
        
        # Clamp the score between 0.1 and 0.9 to avoid extreme confidence
        scam_score = max(0.1, min(0.9, scam_score))
        
        is_scam = scam_score > threshold
        
        # Calculate confidence, adjusted for sensitivity
        # Higher sensitivity leads to higher confidence when classified as scam
        if is_scam:
            # More confident in scam predictions with high sensitivity
            confidence_boost = sensitivity * 10
            confidence = (scam_score * 100) + confidence_boost
            confidence = min(confidence, 99.0)  # Cap at 99%
        else:
            # More confident in legitimate predictions with low sensitivity
            confidence_boost = (1.0 - sensitivity) * 10
            confidence = ((1 - scam_score) * 100) + confidence_boost
            confidence = min(confidence, 99.0)  # Cap at 99%
        
        return is_scam, confidence
    
    def update_with_feedback(self, audio_features, text, user_label):
        """
        Update the model with user feedback.
        
        In a real implementation, this would collect the data for 
        later retraining of the model.
        
        Args:
            audio_features: Extracted audio features
            text: Transcribed text
            user_label: User-provided label (True for scam, False for not scam)
        """
        # This would save the example for later model retraining
        # For demonstration purposes, we just acknowledge it
        print(f"Feedback received: {'Scam' if user_label else 'Not Scam'}")
        print("This data would be saved for model retraining in a real implementation.")
