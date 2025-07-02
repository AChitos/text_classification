import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import yaml
import pickle
import numpy as np
import os

class TextClassificationModel:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the model with configuration."""
        # Handle relative path from src directory
        if not os.path.exists(config_path):
            config_path = os.path.join('..', config_path)
            
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = None
        self.num_classes = None
        self.vocab_size = self.config['model']['vocab_size']
        self.embedding_dim = self.config['model']['embedding_dim']
        self.lstm_units = self.config['model']['lstm_units']
        self.dropout_rate = self.config['model']['dropout_rate']
        self.max_sequence_length = self.config['model']['max_sequence_length']
    
    def build_model(self, num_classes):
        """Build the neural network model."""
        self.num_classes = num_classes
        
        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                name='embedding'
            ),
            
            # Bidirectional LSTM layers
            Bidirectional(LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ), name='bidirectional_lstm_1'),
            
            Bidirectional(LSTM(
                self.lstm_units // 2,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ), name='bidirectional_lstm_2'),
            
            # Dense layers
            Dense(128, activation='relu', name='dense_1'),
            Dropout(self.dropout_rate, name='dropout_1'),
            
            Dense(64, activation='relu', name='dense_2'),
            Dropout(self.dropout_rate, name='dropout_2'),
            
            # Output layer
            Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.config['training']['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_model_summary(self):
        """Get model summary."""
        if self.model is not None:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")
    
    def get_callbacks(self):
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f"{self.config['paths']['models']}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Starting model training...")
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            filepath = f"{self.config['paths']['models']}/text_classification_model.h5"
        
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, filepath=None):
        """Load a trained model."""
        if filepath is None:
            filepath = f"{self.config['paths']['models']}/text_classification_model.h5"
        
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, X):
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_classes(self, X):
        """Predict classes for input data."""
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.predict(X)

def create_simple_model(vocab_size, embedding_dim, max_length, num_classes):
    """Create a simpler model for quick testing."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(64, dropout=0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Example usage
    model_builder = TextClassificationModel()
    
    # Load label encoder to get number of classes
    # Handle path when running from src directory
    label_encoder_path = 'data/processed/label_encoder.pkl'
    if not os.path.exists(label_encoder_path):
        label_encoder_path = os.path.join('..', label_encoder_path)
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    num_classes = len(label_encoder.classes_)
    
    # Build model
    model = model_builder.build_model(num_classes)
    print("Model built successfully!")
    model_builder.get_model_summary()
