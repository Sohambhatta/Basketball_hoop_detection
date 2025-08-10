"""
Model Trainer for Basketball Hoop Detection System
Handles model training, optimization, and validation using PyTorch and Jetson inference
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from PyQt5.QtCore import QThread, pyqtSignal

from config import *
from data_manager import DataManager

class BasketballDataset(Dataset):
    """Custom dataset for basketball images"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.long)

class BasketballNet(nn.Module):
    """CNN model for basketball detection"""
    
    def __init__(self, num_classes=2):
        super(BasketballNet, self).__init__()
        
        # Use ResNet18 as backbone for better performance on Jetson
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.backbone.features(x) if hasattr(self.backbone, 'features') else x
        for name, layer in self.backbone.named_children():
            if name == 'fc':
                break
            x = layer(x)
        
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.backbone.fc(x)
        
        return x

class ModelTrainer(QThread):
    """Model training thread"""
    
    progress_updated = pyqtSignal(int, float, float)  # epoch, train_loss, val_accuracy
    training_completed = pyqtSignal(str, float)  # model_path, final_accuracy
    log_message = pyqtSignal(str)
    
    def __init__(self, data_manager, target_confidence=MIN_CONFIDENCE_THRESHOLD):
        super().__init__()
        self.data_manager = data_manager
        self.target_confidence = target_confidence
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_params = {
            'epochs': DEFAULT_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE
        }
        self.is_training = False
        
    def set_training_params(self, epochs=None, learning_rate=None, batch_size=None):
        """Set training parameters"""
        if epochs is not None:
            self.training_params['epochs'] = epochs
        if learning_rate is not None:
            self.training_params['learning_rate'] = learning_rate
        if batch_size is not None:
            self.training_params['batch_size'] = batch_size
    
    def optimize_parameters(self, X_train, y_train, X_val, y_val):
        """Optimize training parameters to achieve target confidence"""
        self.log_message.emit("Optimizing training parameters...")
        
        # Start with base parameters
        best_params = self.training_params.copy()
        best_accuracy = 0.0
        
        # Test different epoch counts
        epoch_options = [30, 50, 75, 100]
        for epochs in epoch_options:
            if not self.is_training:
                break
                
            self.log_message.emit(f"Testing {epochs} epochs...")
            
            # Quick training test
            test_accuracy = self._quick_train_test(X_train, y_train, X_val, y_val, epochs)
            
            if test_accuracy > best_accuracy and test_accuracy >= self.target_confidence:
                best_accuracy = test_accuracy
                best_params['epochs'] = epochs
            
            # If we've achieved target confidence, try fewer epochs
            if test_accuracy >= self.target_confidence:
                break
        
        self.training_params = best_params
        self.log_message.emit(f"Optimized parameters: {best_params}")
        return best_params
    
    def _quick_train_test(self, X_train, y_train, X_val, y_val, epochs):
        """Quick training test for parameter optimization"""
        # Create smaller subset for quick testing
        subset_size = min(100, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        
        # Create datasets and dataloaders
        train_dataset = BasketballDataset(X_subset, y_subset, self._get_transforms(train=True))
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        val_dataset = BasketballDataset(X_val, y_val, self._get_transforms(train=False))
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create and train model
        model = BasketballNet(num_classes=NUM_CLASSES).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.training_params['learning_rate'])
        
        model.train()
        for epoch in range(min(epochs, 10)):  # Limit to 10 epochs for quick test
            for images, labels in train_loader:
                if not self.is_training:
                    return 0.0
                    
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _get_transforms(self, train=True):
        """Get data transforms"""
        if train:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(MODEL_INPUT_SIZE),
                transforms.RandomRotation(ROTATION_RANGE),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomHorizontalFlip() if HORIZONTAL_FLIP else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(MODEL_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def run(self):
        """Main training loop"""
        try:
            self.is_training = True
            self.log_message.emit("Starting training process...")
            
            # Prepare data
            X_train, X_val, y_train, y_val = self.data_manager.prepare_training_data()
            
            if len(X_train) == 0:
                self.log_message.emit("No training data available!")
                return
            
            self.log_message.emit(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Optimize parameters
            self.optimize_parameters(X_train, y_train, X_val, y_val)
            
            # Create datasets and data loaders
            train_dataset = BasketballDataset(X_train, y_train, self._get_transforms(train=True))
            val_dataset = BasketballDataset(X_val, y_val, self._get_transforms(train=False))
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.training_params['batch_size'], 
                shuffle=True,
                num_workers=2
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.training_params['batch_size'], 
                shuffle=False,
                num_workers=2
            )
            
            # Initialize model
            self.model = BasketballNet(num_classes=NUM_CLASSES).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.training_params['learning_rate'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
            
            # Training loop
            best_val_accuracy = 0.0
            training_history = {'train_loss': [], 'val_accuracy': []}
            
            for epoch in range(self.training_params['epochs']):
                if not self.is_training:
                    break
                
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_samples = 0
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    if not self.is_training:
                        break
                    
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * images.size(0)
                    train_samples += images.size(0)
                
                avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
                
                # Validation phase
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        if not self.is_training:
                            break
                        
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_accuracy = val_correct / val_total if val_total > 0 else 0.0
                
                # Update learning rate
                scheduler.step()
                
                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self._save_model(val_accuracy)
                
                # Log progress
                training_history['train_loss'].append(avg_train_loss)
                training_history['val_accuracy'].append(val_accuracy)
                
                self.progress_updated.emit(epoch + 1, avg_train_loss, val_accuracy)
                self.log_message.emit(
                    f"Epoch {epoch+1}/{self.training_params['epochs']}: "
                    f"Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )
            
            # Save training history
            self._save_training_history(training_history)
            
            if best_val_accuracy >= self.target_confidence:
                self.log_message.emit(f"Training completed! Best accuracy: {best_val_accuracy:.4f}")
                model_path = self._get_model_path()
                self.training_completed.emit(model_path, best_val_accuracy)
            else:
                self.log_message.emit(
                    f"Training completed but target confidence not reached. "
                    f"Best accuracy: {best_val_accuracy:.4f}, Target: {self.target_confidence:.4f}"
                )
                self.training_completed.emit("", best_val_accuracy)
                
        except Exception as e:
            self.log_message.emit(f"Training error: {str(e)}")
        finally:
            self.is_training = False
    
    def stop_training(self):
        """Stop training process"""
        self.is_training = False
    
    def _save_model(self, accuracy):
        """Save trained model"""
        model_path = self._get_model_path()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'training_params': self.training_params,
            'timestamp': datetime.now().isoformat(),
            'model_architecture': 'ResNet18'
        }
        
        torch.save(model_state, model_path)
        return model_path
    
    def _get_model_path(self):
        """Get model save path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(MODELS_DIR, f"basketball_model_{timestamp}.pth")
    
    def _save_training_history(self, history):
        """Save training history"""
        history_path = os.path.join(LOGS_DIR, f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history_path
