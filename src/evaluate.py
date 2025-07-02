import numpy as np
import yaml
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
from model import TextClassificationModel
import pandas as pd

class ModelEvaluator:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the model evaluator."""
        # Handle relative path from src directory
        if not os.path.exists(config_path):
            config_path = os.path.join('..', config_path)
            
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_builder = TextClassificationModel(config_path)
        self.label_encoder = None
        self.tokenizer = None
    
    def load_model_and_preprocessors(self):
        """Load the trained model and preprocessing objects."""
        # Load model
        model_path = f"{self.config['paths']['models']}/text_classification_model.h5"
        self.model_builder.load_model(model_path)
        
        # Load preprocessors
        processed_path = self.config['paths']['processed_data']
        
        with open(f"{processed_path}/label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open(f"{processed_path}/tokenizer.pkl", 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print("Model and preprocessors loaded successfully!")
    
    def load_test_data(self):
        """Load test data."""
        processed_path = self.config['paths']['processed_data']
        
        X_test = np.load(f"{processed_path}/X_test.npy")
        y_test = np.load(f"{processed_path}/y_test.npy")
        
        return X_test, y_test
    
    def evaluate_model(self):
        """Comprehensive model evaluation."""
        print("Starting comprehensive model evaluation...")
        print("="*60)
        
        # Load model and data
        self.load_model_and_preprocessors()
        X_test, y_test = self.load_test_data()
        
        # Make predictions
        print("Making predictions on test set...")
        y_pred_proba = self.model_builder.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print(f"\nOverall Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        # Detailed classification report
        class_names = self.label_encoder.classes_
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, class_names)
        
        # Per-class analysis
        self.analyze_per_class_performance(y_test, y_pred, y_pred_proba, class_names)
        
        # Error analysis
        self.error_analysis(X_test, y_test, y_pred, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Test Set Evaluation', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(self.config['paths']['models'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, 'evaluation_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()
    
    def analyze_per_class_performance(self, y_true, y_pred, y_pred_proba, class_names):
        """Analyze performance for each class."""
        print(f"\nPer-Class Performance Analysis:")
        print("-" * 80)
        
        # Calculate metrics per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(class_names))
        )
        
        # Create DataFrame for better visualization
        results_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Sort by F1-Score
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print(results_df.to_string(index=False, float_format='{:.4f}'.format))
        
        # Plot per-class metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        for i, metric in enumerate(metrics):
            axes[i].bar(range(len(class_names)), results_df[metric], alpha=0.7)
            axes[i].set_title(f'{metric} by Class')
            axes[i].set_xlabel('Classes')
            axes[i].set_ylabel(metric)
            axes[i].set_xticks(range(len(class_names)))
            axes[i].set_xticklabels(class_names, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots_dir = os.path.join(self.config['paths']['models'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, 'per_class_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to: {save_path}")
        plt.show()
        
        # Identify best and worst performing classes
        best_class = results_df.iloc[0]
        worst_class = results_df.iloc[-1]
        
        print(f"\nBest Performing Class: {best_class['Class']}")
        print(f"  - F1-Score: {best_class['F1-Score']:.4f}")
        print(f"  - Precision: {best_class['Precision']:.4f}")
        print(f"  - Recall: {best_class['Recall']:.4f}")
        print(f"  - Support: {int(best_class['Support'])}")
        
        print(f"\nWorst Performing Class: {worst_class['Class']}")
        print(f"  - F1-Score: {worst_class['F1-Score']:.4f}")
        print(f"  - Precision: {worst_class['Precision']:.4f}")
        print(f"  - Recall: {worst_class['Recall']:.4f}")
        print(f"  - Support: {int(worst_class['Support'])}")
    
    def error_analysis(self, X_test, y_true, y_pred, y_pred_proba):
        """Analyze prediction errors."""
        print(f"\nError Analysis:")
        print("-" * 50)
        
        # Find misclassified samples
        misclassified_idx = np.where(y_true != y_pred)[0]
        correct_idx = np.where(y_true == y_pred)[0]
        
        print(f"Total samples: {len(y_true)}")
        print(f"Correctly classified: {len(correct_idx)} ({len(correct_idx)/len(y_true)*100:.2f}%)")
        print(f"Misclassified: {len(misclassified_idx)} ({len(misclassified_idx)/len(y_true)*100:.2f}%)")
        
        if len(misclassified_idx) > 0:
            # Analyze confidence scores
            correct_confidences = np.max(y_pred_proba[correct_idx], axis=1)
            incorrect_confidences = np.max(y_pred_proba[misclassified_idx], axis=1)
            
            print(f"\nConfidence Score Analysis:")
            print(f"Average confidence for correct predictions: {np.mean(correct_confidences):.4f}")
            print(f"Average confidence for incorrect predictions: {np.mean(incorrect_confidences):.4f}")
            
            # Plot confidence distributions
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
            plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Confidence Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            confidence_threshold = np.arange(0.5, 1.0, 0.05)
            coverage = []
            accuracy_at_coverage = []
            
            for threshold in confidence_threshold:
                high_conf_idx = np.where(np.max(y_pred_proba, axis=1) >= threshold)[0]
                if len(high_conf_idx) > 0:
                    coverage.append(len(high_conf_idx) / len(y_true))
                    accuracy_at_coverage.append(
                        accuracy_score(y_true[high_conf_idx], y_pred[high_conf_idx])
                    )
                else:
                    coverage.append(0)
                    accuracy_at_coverage.append(0)
            
            plt.plot(coverage, accuracy_at_coverage, 'bo-')
            plt.xlabel('Coverage (Fraction of Samples)')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Coverage at Different Confidence Thresholds')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots_dir = os.path.join(self.config['paths']['models'], 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, 'error_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error analysis plot saved to: {save_path}")
            plt.show()
            
            # Show some misclassified examples
            self.show_misclassified_examples(misclassified_idx[:5], X_test, y_true, y_pred, y_pred_proba)
    
    def show_misclassified_examples(self, indices, X_test, y_true, y_pred, y_pred_proba):
        """Show examples of misclassified samples."""
        print(f"\nMisclassified Examples:")
        print("-" * 60)
        
        class_names = self.label_encoder.classes_
        
        for i, idx in enumerate(indices):
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            confidence = np.max(y_pred_proba[idx])
            
            # Reconstruct text from tokens (simplified)
            tokens = X_test[idx]
            non_zero_tokens = tokens[tokens != 0]
            
            print(f"Example {i+1}:")
            print(f"  True Class: {true_class}")
            print(f"  Predicted Class: {pred_class}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Token sequence length: {len(non_zero_tokens)}")
            print()

def main():
    """Main evaluation function."""
    print("Starting Model Evaluation")
    print("="*40)
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model()
    
    print(f"\nEvaluation completed!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
