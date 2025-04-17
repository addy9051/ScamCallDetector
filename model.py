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
        
        # Comprehensive list of Hindi keywords based on real scam calls
        self.hindi_keywords = {
            "scam": [
                "धोखा", "फ्रॉड", "तुरंत", "जल्दी", "पैसा", "नकद", "खाता", "पासवर्ड",
                "पुलिस", "अपराध", "जुर्माना", "पैकेज", "कूरियर", "आरोप", "गिरफ्तार", 
                "कानूनी", "सरकारी", "दस्तावेज़", "पार्सल", "सील", "जब्त"
            ],
            "safe": ["धन्यवाद", "सहायता", "समर्थन", "जानकारी", "आभार", "कैसे हैं आप"]
        }
        
        # Enhanced scam keywords based on common Hindi scam patterns
        # Includes specific terms used in courier/package, banking, and police scams
        self.scam_keyword_weights = {
            # General scam terms
            "धोखा": 0.8,      # fraud
            "फ्रॉड": 0.8,      # fraud
            "तुरंत": 0.5,      # immediate/urgent
            "जल्दी": 0.5,      # hurry
            "अभी": 0.5,        # now
            
            # Financial terms
            "पैसा": 0.4,       # money
            "नकद": 0.6,        # cash
            "खाता": 0.5,       # account
            "बैंक": 0.4,       # bank
            "कार्ड": 0.6,      # card
            "ओटीपी": 0.8,      # OTP
            "पैसे": 0.5,       # money (plural)
            "रुपए": 0.4,       # rupees
            "आधार": 0.6,       # Aadhaar (ID)
            "पैन": 0.6,        # PAN (ID)
            
            # Courier/Package scam terms
            "पार्सल": 0.7,     # parcel
            "पैकेज": 0.7,      # package
            "कूरियर": 0.7,     # courier
            "डिलीवरी": 0.6,    # delivery
            "सामान": 0.5,      # goods/items
            "भेजा": 0.4,       # sent
            "फेडेक्स": 0.7,    # FedEx
            "डीटीडीसी": 0.7,   # DTDC
            
            # Police/legal scam terms
            "पुलिस": 0.8,      # police
            "अपराध": 0.8,      # crime
            "आरोप": 0.8,       # accusation/charge
            "जुर्माना": 0.8,   # fine/penalty
            "कानूनी": 0.7,     # legal
            "कार्रवाई": 0.7,   # action/proceedings
            "गिरफ्तार": 0.9,   # arrest
            "जांच": 0.6,       # investigation
            "अधिकारी": 0.6,    # officer
            "सरकारी": 0.6,     # government
            "दस्तावेज़": 0.6,   # documents
            "सील": 0.7,        # seal
            "जब्त": 0.8,       # confiscate/seize
            
            # Security/credentials
            "पासवर्ड": 0.8,    # password
            "लॉगिन": 0.7,      # login
            "गोपनीय": 0.6,     # confidential
            "सत्यापन": 0.6,    # verification
            
            # Other scam indicators
            "लॉटरी": 0.8,      # lottery
            "इनाम": 0.7,       # prize
            "जीता": 0.7,       # won
            "मुफ्त": 0.7       # free
        }
        
        # Safe keywords with negative weights (reduce scam score)
        self.safe_keyword_weights = {
            "धन्यवाद": -0.4,   # thank you
            "सहायता": -0.3,    # help
            "समर्थन": -0.3,    # support
            "जानकारी": -0.2,   # information
            "आभार": -0.4,      # gratitude
            "स्वागत": -0.3,    # welcome
            "कैसे हैं आप": -0.5, # how are you
            "शुभकामनाएं": -0.4  # best wishes
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
        # Start with a higher base score for more aggressive scam detection
        base_score_adjustment = (sensitivity - 0.5) * 0.3  # Increased impact of sensitivity
        scam_score = 0.4 + base_score_adjustment  # Higher starting point
        
        # Analyze text if available
        found_keywords = []
        keyword_count = 0
        courier_found = False
        police_found = False
        
        if text is not None and text:
            # Analyze text for scam keywords
            for keyword, weight in self.scam_keyword_weights.items():
                if keyword in text:
                    # Add to found keywords list for reporting
                    found_keywords.append(keyword)
                    keyword_count += 1
                    
                    # Adjust keyword weight based on sensitivity
                    adjusted_weight = weight * (0.8 + sensitivity * 0.4)
                    
                    # Increase impact of keywords
                    scam_score += adjusted_weight * 0.15  # Increased from 0.1
            
            # Special pattern detection for courier/package scams
            courier_terms = ["पार्सल", "पैकेज", "कूरियर", "डिलीवरी", "फेडेक्स", "डीटीडीसी"]
            police_terms = ["पुलिस", "अपराध", "आरोप", "कानूनी", "गिरफ्तार"]
            
            # Check for combined patterns (courier + police, strong indicator of scam)
            courier_found = any(term in text for term in courier_terms)
            police_found = any(term in text for term in police_terms)
            
            if courier_found and police_found:
                # Strong boost for Fedex/courier/police scam pattern
                scam_score += 0.3 * sensitivity
            
            # Check for safe keywords that might reduce the scam score
            safe_keyword_count = 0
            for keyword, weight in self.safe_keyword_weights.items():
                if keyword in text:
                    safe_keyword_count += 1
                    # For safe keywords, lower sensitivity reduces their impact
                    # (making the model more cautious)
                    discount_factor = 1.0 - (sensitivity * 0.5)  # Increased discount
                    scam_score += weight * 0.08 * discount_factor  # Reduced impact
            
            # If we have multiple scam keywords and no safe keywords, boost the score
            if keyword_count >= 3 and safe_keyword_count == 0:
                scam_score += 0.15 * sensitivity
        
        # Analyze audio features
        # In a real model, we would analyze specific patterns in the audio features
        # For this demo, we'll use improved heuristics
        
        # 1. Check for rapid speech (a common scam indicator)
        if len(audio_features) > 10:
            feature_avg = np.mean(audio_features[:10])
            # Scale this to a small contribution to the score, influenced by sensitivity
            scam_score += (feature_avg - np.mean(audio_features)) * 0.015 * (1.0 + sensitivity)
        
        # 2. Check the variance of features as a proxy for speech variability
        # Scammers often have more variable speech patterns
        feature_variance = np.var(audio_features)
        variance_threshold = 0.9 - (sensitivity * 0.6)  # Lower threshold with higher sensitivity
        if feature_variance > variance_threshold:
            scam_score += 0.08 * (1.0 + sensitivity)  # Increased impact
        
        # 3. Check for extreme values in audio features
        # Scammers may have more extreme emotional patterns
        if len(audio_features) > 0:
            max_feature = np.max(np.abs(audio_features))
            if max_feature > 4.0:  # Lower threshold to catch more
                scam_score += 0.07 * sensitivity  # Increased impact
        
        # 4. Check for pitch variation (new feature)
        # High pitch variation is common in scam calls to create urgency
        if len(audio_features) > 100:
            # Use a segment of features that might correspond to pitch information
            pitch_indicators = audio_features[30:50]  # Approximate range where pitch info might be
            pitch_variation = np.std(pitch_indicators)
            if pitch_variation > 1.0:
                scam_score += 0.1 * sensitivity
        
        # Add minimal randomness (reduced significantly)
        # randomness_factor = 0.05 * (1.0 - sensitivity * 0.8)  # Greatly reduced randomness
        # scam_score += (random.random() - 0.5) * randomness_factor
        
        # Adjust threshold based on sensitivity
        # Higher sensitivity means lower threshold to classify as scam
        threshold = 0.48 - (sensitivity - 0.5) * 0.25  # Lower baseline threshold
        
        # Clamp the score between 0.1 and 0.9 to avoid extreme confidence
        scam_score = max(0.1, min(0.9, scam_score))
        
        is_scam = scam_score > threshold
        
        # Calculate confidence, adjusted for sensitivity
        # Higher sensitivity leads to higher confidence when classified as scam
        if is_scam:
            # More confident in scam predictions with high sensitivity
            confidence_boost = sensitivity * 12  # Increased boost
            confidence = (scam_score * 100) + confidence_boost
            confidence = min(confidence, 99.0)  # Cap at 99%
        else:
            # More confident in legitimate predictions with low sensitivity
            confidence_boost = (1.0 - sensitivity) * 10
            confidence = ((1 - scam_score) * 100) + confidence_boost
            confidence = min(confidence, 99.0)  # Cap at 99%
        
        # Return a more comprehensive result dictionary
        return {
            "is_scam": is_scam,
            "confidence": confidence,
            "detected_keywords": found_keywords if len(found_keywords) > 0 else [],
            "scam_patterns": {
                "courier_police_pattern": courier_found and police_found,
                "multiple_scam_keywords": keyword_count >= 3,
                "high_urgency": any(k in text for k in ["तुरंत", "जल्दी", "अभी"]) if text else False
            } if text else {}
        }
    
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
