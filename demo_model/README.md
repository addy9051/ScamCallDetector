# Demo Model Directory

This directory would contain the trained models in a production environment.

In a complete implementation, this directory would include:

1. `audio_model/` - A TensorFlow model trained on audio features
2. `text_model/` - A TensorFlow model trained on text features
3. `fusion_model/` - A model that combines outputs from the audio and text models
4. `hindi_keywords.pkl` - A dictionary of Hindi keywords related to scams
5. `tokenizer.pkl` - A tokenizer for processing Hindi text

## Model Training Approach

The models would be trained using:

1. **TeleAntiFraud-28k dataset** - Hindi samples extracted using language detection
2. **HAV-DF dataset** - Synthetic Hindi deepfakes to cover voice cloning scams
3. **Ethically scraped data** - From publicly available sources like YouTube

## Model Architecture

- **Audio Model**: CNN or RNN architecture processing MFCCs and other acoustic features
- **Text Model**: Transformer-based model (e.g., BERT for Hindi) processing transcribed text
- **Fusion Model**: Neural network that combines predictions from both models

## Training Process

1. Extract Hindi samples from TeleAntiFraud-28k
2. Augment with synthetic data from HAV-DF
3. Add ethically scraped examples
4. Extract audio and text features
5. Train separate audio and text models
6. Train fusion model on combined outputs
7. Evaluate on held-out test set
8. Deploy best-performing model

## Continuous Improvement

In production, this model would be regularly updated with:
- User-reported scam examples
- New scam tactics
- Performance improvements

The final system would achieve high accuracy in identifying Hindi-specific scam calls while minimizing false positives.
