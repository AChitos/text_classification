"""
Prediction/Inference script for the text classification model.
This script loads a trained model and makes predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import re
import yaml
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MaintenanceReportClassifier:
    """
    A classifier for predicting categories from text reports.
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize the classifier.
        
        Args:
            config_path (str): Path to the configuration file
        """
        # Handle relative path from src directory
        if not os.path.exists(config_path):
            config_path = os.path.join('..', config_path)
            
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_sequence_length = self.config['model']['max_sequence_length']
        
        # Load model and preprocessors
        self.load_model_and_preprocessors()
            
    def load_model_and_preprocessors(self):
        """Load the trained model and preprocessing objects."""
        try:
            # Load model
            model_path = f"{self.config['paths']['models']}/text_classification_model.h5"
            if not os.path.exists(model_path):
                model_path = f"{self.config['paths']['models']}/best_model.h5"
            
            self.model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            
            # Load preprocessors
            processed_path = self.config['paths']['processed_data']
            
            with open(f"{processed_path}/label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open(f"{processed_path}/tokenizer.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            print("Preprocessors loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model or preprocessors: {e}")
            print("Please ensure the model is trained and saved properly.")
            raise
            
    def clean_text(self, text):
        """Clean and normalize text data."""
        if not text or text.strip() == "":
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_text(self, text):
        """Preprocess a single text for prediction."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            raise ValueError("Text is empty after cleaning. Please provide valid text.")
        
        # Tokenize
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        
        # Pad sequence
        padded_sequence = pad_sequences(
            sequence,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        return padded_sequence
            
    def predict_single(self, description):
        """Predict for a single description."""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(description)
            
            # Make prediction
            prediction_proba = self.model.predict(processed_text, verbose=0)
            predicted_class_idx = np.argmax(prediction_proba, axis=1)[0]
            confidence = np.max(prediction_proba)
            
            # Convert to class name
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': {
                    self.label_encoder.classes_[i]: float(prediction_proba[0][i])
                    for i in range(len(self.label_encoder.classes_))
                }
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def predict_batch(self, descriptions):
        """Predict problem types for multiple descriptions."""
        results = []
        
        for i, description in enumerate(descriptions):
            print(f"Processing description {i+1}/{len(descriptions)}...")
            result = self.predict_single(description)
            if result:
                result['input_description'] = description
                results.append(result)
            else:
                results.append({
                    'input_description': description,
                    'predicted_class': 'Error',
                    'confidence': 0.0,
                    'all_probabilities': {}
                })
        
        return results
    
    def get_top_predictions(self, description, top_k=3):
        """Get top-k predictions for a description."""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(description)
            
            # Make prediction
            prediction_proba = self.model.predict(processed_text, verbose=0)[0]
            
            # Get top-k predictions
            top_indices = np.argsort(prediction_proba)[-top_k:][::-1]
            
            top_predictions = []
            for idx in top_indices:
                class_name = self.label_encoder.classes_[idx]
                probability = prediction_proba[idx]
                top_predictions.append({
                    'class': class_name,
                    'probability': float(probability)
                })
            
            return top_predictions
            
        except Exception as e:
            print(f"Error getting top predictions: {e}")
            return []

def main():
    """
    Main function to demonstrate the prediction functionality.
    """
    # Initialize classifier
    print("Loading model and initializing classifier...")
    try:
        classifier = MaintenanceReportClassifier()
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        print("Please ensure the model is trained. Run: python src/train.py")
        return
    
    # Example predictions
    sample_reports = [
        "The screw is too small for the hole",
        "Motor is overheating and making noise",
        "Hydraulic pressure is too low",
        "Electrical connection is faulty",
        "Sensor is not responding correctly",
        "The bearing is worn out and noisy"
    ]
    
    print("\n" + "="*70)
    print("MAINTENANCE REPORT CLASSIFICATION DEMO")
    print("="*70)
    
    for i, report in enumerate(sample_reports, 1):
        print(f"\nReport {i}:")
        print(f"Text: {report}")
        
        result = classifier.predict_single(report)
        
        if result:
            print(f"Predicted Category: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            # Show top 3 categories
            top_3 = classifier.get_top_predictions(report, top_k=3)
            print("Top 3 predictions:")
            for j, pred in enumerate(top_3, 1):
                print(f"  {j}. {pred['class']}: {pred['probability']:.3f}")
        else:
            print("Error making prediction")
        print("-" * 50)
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE PREDICTION MODE")
    print("Enter maintenance reports to classify (type 'quit', 'exit' or 'q' to exit)")
    print("="*70)
    
    while True:
        user_input = input("\nEnter a maintenance report: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_input:
            print("Please enter a valid maintenance report.")
            continue
        
        try:
            result = classifier.predict_single(user_input)
            
            if result:
                print(f"\nPredicted Category: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.3f}")
                
                # Show top 3 categories
                top_3 = classifier.get_top_predictions(user_input, top_k=3)
                print("Top 3 predictions:")
                for j, pred in enumerate(top_3, 1):
                    print(f"  {j}. {pred['class']}: {pred['probability']:.3f}")
            else:
                print("Error making prediction")
                
        except Exception as e:
            print(f"Error making prediction: {e}")
    
    print("\nThank you for using the Maintenance Report Classifier!")

if __name__ == "__main__":
    main()
