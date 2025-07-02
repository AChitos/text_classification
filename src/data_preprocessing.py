import pandas as pd
import numpy as np
import re
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

class DataPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the data preprocessor with configuration."""
        # Handle relative path from src directory
        if not os.path.exists(config_path):
            config_path = os.path.join('..', config_path)
            
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.tokenizer = None
        self.label_encoder = None
        self.max_sequence_length = self.config['model']['max_sequence_length']
        self.vocab_size = self.config['model']['vocab_size']
    
    def clean_text(self, text):
        """Clean and normalize text data."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def load_data(self, file_path=None):
        """Load data from CSV file."""
        if file_path is None:
            file_path = self.config['paths']['raw_data']
        
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}. Please check the path.")
            raise
    
    def preprocess_data(self, df):
        """Preprocess the dataframe."""
        print("Starting data preprocessing...")
        
        # Check required columns
        required_columns = ['description', 'problem_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean text data
        df['cleaned_description'] = df['description'].apply(self.clean_text)
        
        # Remove empty descriptions
        df = df[df['cleaned_description'].str.len() > 0].copy()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        df['encoded_label'] = self.label_encoder.fit_transform(df['problem_type'])
        
        print(f"Number of unique problem types: {len(self.label_encoder.classes_)}")
        print(f"Problem types: {list(self.label_encoder.classes_)}")
        
        return df
    
    def tokenize_text(self, texts):
        """Tokenize and pad text sequences."""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(
                num_words=self.vocab_size,
                oov_token='<OOV>'
            )
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        return padded_sequences
    
    def split_data(self, df):
        """Split data into train, validation, and test sets."""
        X = df['cleaned_description'].values
        y = df['encoded_label'].values
        
        # First split: train + val, test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Second split: train, val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['training']['validation_split'],
            random_state=self.config['data']['random_state'],
            stratify=y_temp
        )
        
        # Tokenize texts
        X_train_seq = self.tokenize_text(X_train)
        X_val_seq = self.tokenize_text(X_val)
        X_test_seq = self.tokenize_text(X_test)
        
        return (X_train_seq, X_val_seq, X_test_seq), (y_train, y_val, y_test)
    
    def save_processed_data(self, data_splits, label_splits):
        """Save processed data and preprocessing objects."""
        X_train, X_val, X_test = data_splits
        y_train, y_val, y_test = label_splits
        
        # Create processed data directory
        os.makedirs(self.config['paths']['processed_data'], exist_ok=True)
        
        # Save data splits
        np.save(f"{self.config['paths']['processed_data']}/X_train.npy", X_train)
        np.save(f"{self.config['paths']['processed_data']}/X_val.npy", X_val)
        np.save(f"{self.config['paths']['processed_data']}/X_test.npy", X_test)
        np.save(f"{self.config['paths']['processed_data']}/y_train.npy", y_train)
        np.save(f"{self.config['paths']['processed_data']}/y_val.npy", y_val)
        np.save(f"{self.config['paths']['processed_data']}/y_test.npy", y_test)
        
        # Save preprocessing objects
        with open(f"{self.config['paths']['processed_data']}/tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        with open(f"{self.config['paths']['processed_data']}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print("Processed data saved successfully!")
        
        # Print data shapes
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        # Load data
        df = self.load_data()
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Split data
        data_splits, label_splits = self.split_data(df_processed)
        
        # Save processed data
        self.save_processed_data(data_splits, label_splits)
        
        return data_splits, label_splits

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data_splits, label_splits = preprocessor.run_preprocessing()
    print("Data preprocessing completed successfully!")
