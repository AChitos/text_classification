#!/usr/bin/env python3
"""
Model Evaluation Script
Loads the trained model and demonstrates its performance on test data.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Load configuration
config_path = 'config/config.yaml'
if not os.path.exists(config_path):
    config_path = os.path.join('..', config_path)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def load_test_data():
    """Load preprocessed test data"""
    try:
        processed_data_path = config['paths']['processed_data']
        
        # Load test data (numpy arrays)
        X_test = np.load(os.path.join(processed_data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(processed_data_path, 'y_test.npy'))
        
        # Load label encoder
        with open(os.path.join(processed_data_path, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
            
        print(f"✓ Test data loaded successfully")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - Number of classes: {len(label_encoder.classes_)}")
        print(f"  - Classes: {list(label_encoder.classes_)}")
        
        return X_test, y_test, label_encoder
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return None, None, None

def load_trained_model():
    """Load the trained model"""
    try:
        model_path = os.path.join(config['paths']['models'], 'best_model.h5')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"✓ Model loaded successfully from {model_path}")
            return model
        else:
            print(f"✗ Model file not found at {model_path}")
            return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def evaluate_model_performance(model, X_test, y_test, label_encoder):
    """Evaluate model performance and display results"""
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Make predictions
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"\n🎯 Test Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print(f"\n📊 Detailed Classification Report:")
    print("-" * 50)
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Display confusion matrix as heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Text Classification Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix plot
    plot_path = os.path.join(config['paths']['models'], 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Confusion matrix saved to: {plot_path}")
    plt.show()
    
    return accuracy, report

def demonstrate_predictions(model, X_test, y_test, label_encoder, num_examples=5):
    """Demonstrate model predictions on sample texts"""
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Load original text data to show actual descriptions
    try:
        with open(os.path.join(config['paths']['processed_data'], 'tokenizer.pkl'), 'rb') as f:
            tokenizer = pickle.load(f)
    except:
        print("Could not load tokenizer for text demonstration")
        return
    
    # Get random sample indices
    sample_indices = np.random.choice(len(X_test), num_examples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Get prediction
        prediction = model.predict(X_test[idx:idx+1])
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = label_encoder.classes_[predicted_class_idx]
        actual_class = label_encoder.classes_[y_test[idx]]
        confidence = np.max(prediction[0]) * 100
        
        # Try to convert back to text (simplified)
        sequence = X_test[idx]
        words = []
        for token_id in sequence:
            if token_id > 0:  # Skip padding
                for word, id in tokenizer.word_index.items():
                    if id == token_id:
                        words.append(word)
                        break
        text_approximation = ' '.join(words[:10]) + "..." if len(words) > 10 else ' '.join(words)
        
        print(f"\n📝 Example {i+1}:")
        print(f"   Text (partial): {text_approximation}")
        print(f"   Actual: {actual_class}")
        print(f"   Predicted: {predicted_class}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   {'✓ CORRECT' if predicted_class == actual_class else '✗ INCORRECT'}")

def main():
    """Main evaluation function"""
    print("🚀 Starting Model Evaluation")
    print("="*60)
    
    # Load test data
    X_test, y_test, label_encoder = load_test_data()
    if X_test is None:
        return
    
    # Load trained model
    model = load_trained_model()
    if model is None:
        return
    
    # Model summary
    print(f"\n🏗️ Model Architecture:")
    print("-" * 30)
    model.summary()
    
    # Evaluate performance
    accuracy, report = evaluate_model_performance(model, X_test, y_test, label_encoder)
    
    # Demonstrate predictions
    demonstrate_predictions(model, X_test, y_test, label_encoder)
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"   Final Test Accuracy: {accuracy:.2f}%")
    print(f"   Model performs {'EXCELLENTLY' if accuracy >= 95 else 'WELL' if accuracy >= 85 else 'ADEQUATELY' if accuracy >= 75 else 'POORLY'}")

if __name__ == "__main__":
    main()
