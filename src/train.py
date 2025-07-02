import numpy as np
import yaml
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from model import TextClassificationModel
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class ModelTrainer:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the model trainer."""
        # Handle relative path from src directory
        if not os.path.exists(config_path):
            config_path = os.path.join('..', config_path)
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_builder = TextClassificationModel(config_path)
        self.label_encoder = None
        self.history = None
    
    def load_processed_data(self):
        """Load preprocessed data."""
        processed_path = self.config['paths']['processed_data']
        
        # Load data splits
        X_train = np.load(f"{processed_path}/X_train.npy")
        X_val = np.load(f"{processed_path}/X_val.npy")
        X_test = np.load(f"{processed_path}/X_test.npy")
        y_train = np.load(f"{processed_path}/y_train.npy")
        y_val = np.load(f"{processed_path}/y_val.npy")
        y_test = np.load(f"{processed_path}/y_test.npy")
        
        # Load label encoder
        with open(f"{processed_path}/label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"Data loaded successfully!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def train_model(self):
        """Train the text classification model."""
        # Load data
        (X_train, X_val, X_test), (y_train, y_val, y_test) = self.load_processed_data()
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        model = self.model_builder.build_model(num_classes)
        
        # Display model summary
        print("\nModel Architecture:")
        self.model_builder.get_model_summary()
        
        # Train model
        print("\nStarting training...")
        self.history = self.model_builder.train(X_train, y_train, X_val, y_val)
        
        # Save model
        os.makedirs(self.config['paths']['models'], exist_ok=True)
        self.model_builder.save_model()
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return self.history, (test_loss, test_accuracy)
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot training & validation loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(self.config['paths']['models'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_path = os.path.join(plots_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {plot_path}")
        plt.show()
    
    def generate_detailed_report(self):
        """Generate detailed training and evaluation report."""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        # Load test data for evaluation
        (X_train, X_val, X_test), (y_train, y_val, y_test) = self.load_processed_data()
        
        # Make predictions
        y_pred = self.model_builder.predict_classes(X_test)
        y_pred_proba = self.model_builder.predict_proba(X_test)
        
        # Classification report
        print("\n" + "="*80)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*80)
        
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # Print classification report
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(self.config['paths']['models'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        confusion_matrix_path = os.path.join(plots_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {confusion_matrix_path}")
        plt.show()
        
        # Training summary
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        print(f"\nTraining Summary:")
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        
        # Per-class performance
        print(f"\nPer-Class Performance:")
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1_score = report[class_name]['f1-score']
                support = report[class_name]['support']
                print(f"{class_name:20} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}, Support: {support}")
        
        # Overall metrics
        accuracy = report['accuracy']
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']
        
        print(f"\nOverall Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro Average - Precision: {macro_avg['precision']:.4f}, Recall: {macro_avg['recall']:.4f}, F1: {macro_avg['f1-score']:.4f}")
        print(f"Weighted Average - Precision: {weighted_avg['precision']:.4f}, Recall: {weighted_avg['recall']:.4f}, F1: {weighted_avg['f1-score']:.4f}")

def main():
    """Main training function."""
    print("Starting Text Classification Model Training")
    print("="*50)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train model
    history, test_results = trainer.train_model()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Generate detailed report
    trainer.generate_detailed_report()
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {trainer.config['paths']['models']}")

if __name__ == "__main__":
    main()
