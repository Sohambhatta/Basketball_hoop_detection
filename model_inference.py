"""
Model Inference for Basketball Hoop Detection System
Handles real-time inference and prediction using trained models
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from datetime import datetime
import json

from model_trainer import BasketballNet
from config import MODELS_DIR, MODEL_INPUT_SIZE, MIN_CONFIDENCE_THRESHOLD

class ModelInference:
    """Handles model inference for real-time detection"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = self._get_inference_transform()
        self.model_info = None
        self.confidence_threshold = MIN_CONFIDENCE_THRESHOLD
        
        # Statistics tracking
        self.stats = {
            'total_predictions': 0,
            'made_predictions': 0,
            'missed_predictions': 0,
            'high_confidence_predictions': 0,
            'start_time': None
        }
    
    def _get_inference_transform(self):
        """Get preprocessing transforms for inference"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load trained model from file"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model state
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            self.model = BasketballNet(num_classes=2).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Store model info
            self.model_info = {
                'accuracy': checkpoint.get('accuracy', 0.0),
                'training_params': checkpoint.get('training_params', {}),
                'timestamp': checkpoint.get('timestamp', ''),
                'architecture': checkpoint.get('model_architecture', 'ResNet18')
            }
            
            print(f"Model loaded successfully. Accuracy: {self.model_info['accuracy']:.4f}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_latest_model(self):
        """Get the path to the latest trained model"""
        if not os.path.exists(MODELS_DIR):
            return None
        
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
        if not model_files:
            return None
        
        # Sort by modification time
        model_paths = [os.path.join(MODELS_DIR, f) for f in model_files]
        latest_model = max(model_paths, key=os.path.getmtime)
        
        return latest_model
    
    def predict(self, image, return_confidence=True):
        """Make prediction on single image"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume RGB
                    pass
                else:
                    raise ValueError(f"Unsupported image shape: {image.shape}")
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                confidence_value = confidence.item()
                predicted_class = predicted.item()
            
            # Update statistics
            self.stats['total_predictions'] += 1
            if predicted_class == 1:  # Made shot
                self.stats['made_predictions'] += 1
            else:  # Missed shot
                self.stats['missed_predictions'] += 1
            
            if confidence_value >= self.confidence_threshold:
                self.stats['high_confidence_predictions'] += 1
            
            # Return results
            result = {
                'prediction': predicted_class,  # 0: missed, 1: made
                'confidence': confidence_value,
                'is_made': predicted_class == 1,
                'meets_threshold': confidence_value >= self.confidence_threshold,
                'probabilities': {
                    'missed': probabilities[0][0].item(),
                    'made': probabilities[0][1].item()
                }
            }
            
            if return_confidence:
                return result
            else:
                return result['prediction'], result['confidence']
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def predict_batch(self, images):
        """Make predictions on batch of images"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            # Preprocess images
            input_tensors = []
            for image in images:
                if isinstance(image, np.ndarray):
                    if image.shape[2] == 4:  # RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
                input_tensor = self.transform(image)
                input_tensors.append(input_tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(input_tensors).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
            
            # Process results
            results = []
            for i in range(len(images)):
                confidence_value = confidences[i].item()
                predicted_class = predicted[i].item()
                
                result = {
                    'prediction': predicted_class,
                    'confidence': confidence_value,
                    'is_made': predicted_class == 1,
                    'meets_threshold': confidence_value >= self.confidence_threshold,
                    'probabilities': {
                        'missed': probabilities[i][0].item(),
                        'made': probabilities[i][1].item()
                    }
                }
                results.append(result)
                
                # Update statistics
                self.stats['total_predictions'] += 1
                if predicted_class == 1:
                    self.stats['made_predictions'] += 1
                else:
                    self.stats['missed_predictions'] += 1
                
                if confidence_value >= self.confidence_threshold:
                    self.stats['high_confidence_predictions'] += 1
            
            return results
            
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            return None
    
    def test_model_confidence(self, test_images, test_labels):
        """Test model confidence on validation set"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            correct_predictions = 0
            high_confidence_correct = 0
            total_samples = len(test_images)
            
            results = self.predict_batch(test_images)
            
            for i, result in enumerate(results):
                true_label = test_labels[i]
                predicted_label = result['prediction']
                confidence = result['confidence']
                
                if predicted_label == true_label:
                    correct_predictions += 1
                    if confidence >= self.confidence_threshold:
                        high_confidence_correct += 1
            
            overall_accuracy = correct_predictions / total_samples
            high_confidence_accuracy = high_confidence_correct / max(1, sum(1 for r in results if r['meets_threshold']))
            
            test_results = {
                'overall_accuracy': overall_accuracy,
                'high_confidence_accuracy': high_confidence_accuracy,
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'high_confidence_predictions': sum(1 for r in results if r['meets_threshold']),
                'confidence_threshold': self.confidence_threshold,
                'average_confidence': np.mean([r['confidence'] for r in results])
            }
            
            return test_results
            
        except Exception as e:
            print(f"Error during model testing: {e}")
            return None
    
    def reset_stats(self):
        """Reset prediction statistics"""
        self.stats = {
            'total_predictions': 0,
            'made_predictions': 0,
            'missed_predictions': 0,
            'high_confidence_predictions': 0,
            'start_time': datetime.now()
        }
    
    def get_stats(self):
        """Get current prediction statistics"""
        stats = self.stats.copy()
        
        if stats['total_predictions'] > 0:
            stats['made_percentage'] = (stats['made_predictions'] / stats['total_predictions']) * 100
            stats['missed_percentage'] = (stats['missed_predictions'] / stats['total_predictions']) * 100
            stats['high_confidence_percentage'] = (stats['high_confidence_predictions'] / stats['total_predictions']) * 100
        else:
            stats['made_percentage'] = 0.0
            stats['missed_percentage'] = 0.0
            stats['high_confidence_percentage'] = 0.0
        
        if stats['start_time']:
            elapsed = datetime.now() - stats['start_time']
            stats['elapsed_time'] = elapsed.total_seconds()
            if stats['elapsed_time'] > 0:
                stats['predictions_per_second'] = stats['total_predictions'] / stats['elapsed_time']
            else:
                stats['predictions_per_second'] = 0.0
        
        return stats
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def get_model_info(self):
        """Get loaded model information"""
        return self.model_info
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
    
    def export_stats(self, filepath):
        """Export statistics to file"""
        stats = self.get_stats()
        stats['model_info'] = self.model_info
        stats['export_timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        return filepath
