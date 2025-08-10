"""
Data Manager for Basketball Hoop Detection System
Handles dataset creation, augmentation, and management
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
import random
from config import DATASETS_DIR, MODEL_INPUT_SIZE, VALIDATION_SPLIT

class DataManager:
    """Manages training data collection and preprocessing"""
    
    def __init__(self, dataset_name="basketball_dataset"):
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(DATASETS_DIR, dataset_name)
        self.made_path = os.path.join(self.dataset_path, "made")
        self.missed_path = os.path.join(self.dataset_path, "missed")
        self.metadata_path = os.path.join(self.dataset_path, "metadata.json")
        
        self.create_dataset_structure()
        self.metadata = self.load_metadata()
    
    def create_dataset_structure(self):
        """Create dataset directory structure"""
        os.makedirs(self.made_path, exist_ok=True)
        os.makedirs(self.missed_path, exist_ok=True)
    
    def load_metadata(self):
        """Load or create dataset metadata"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "created": datetime.now().isoformat(),
                "made_count": 0,
                "missed_count": 0,
                "total_images": 0,
                "last_updated": datetime.now().isoformat()
            }
    
    def save_metadata(self):
        """Save metadata to file"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_image(self, image, label):
        """Add image to dataset with label (True for made, False for missed)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        if label:
            filename = f"made_{timestamp}.jpg"
            filepath = os.path.join(self.made_path, filename)
            self.metadata["made_count"] += 1
        else:
            filename = f"missed_{timestamp}.jpg"
            filepath = os.path.join(self.missed_path, filename)
            self.metadata["missed_count"] += 1
        
        # Resize image to model input size
        image_resized = cv2.resize(image, MODEL_INPUT_SIZE)
        cv2.imwrite(filepath, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
        
        self.metadata["total_images"] += 1
        self.save_metadata()
        
        return filepath
    
    def get_dataset_stats(self):
        """Get current dataset statistics"""
        made_files = len([f for f in os.listdir(self.made_path) if f.endswith('.jpg')])
        missed_files = len([f for f in os.listdir(self.missed_path) if f.endswith('.jpg')])
        
        return {
            "made_count": made_files,
            "missed_count": missed_files,
            "total_images": made_files + missed_files,
            "balance_ratio": made_files / max(missed_files, 1)
        }
    
    def augment_image(self, image):
        """Apply data augmentation to image"""
        pil_image = Image.fromarray(image)
        
        # Random rotation
        angle = random.uniform(-15, 15)
        pil_image = pil_image.rotate(angle)
        
        # Random brightness
        brightness = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness)
        
        # Random horizontal flip
        if random.random() > 0.5:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random zoom (crop and resize)
        if random.random() > 0.5:
            width, height = pil_image.size
            zoom_factor = random.uniform(0.9, 1.1)
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            
            if zoom_factor > 1:
                # Crop
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                pil_image = pil_image.crop((left, top, left + width, top + height))
            else:
                # Resize and pad
                pil_image = pil_image.resize((new_width, new_height))
                new_img = Image.new('RGB', (width, height), (0, 0, 0))
                new_img.paste(pil_image, ((width - new_width) // 2, (height - new_height) // 2))
                pil_image = new_img
        
        return np.array(pil_image)
    
    def augment_dataset(self, target_per_class=None):
        """Augment dataset to balance classes and increase size"""
        stats = self.get_dataset_stats()
        
        if target_per_class is None:
            target_per_class = max(stats["made_count"], stats["missed_count"]) * 2
        
        # Augment made images
        made_images_needed = target_per_class - stats["made_count"]
        if made_images_needed > 0:
            self._augment_class(self.made_path, made_images_needed, True)
        
        # Augment missed images
        missed_images_needed = target_per_class - stats["missed_count"]
        if missed_images_needed > 0:
            self._augment_class(self.missed_path, missed_images_needed, False)
    
    def _augment_class(self, class_path, num_augmentations, is_made_class):
        """Augment a specific class"""
        existing_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        
        for i in range(num_augmentations):
            # Pick random existing image
            source_file = random.choice(existing_files)
            source_path = os.path.join(class_path, source_file)
            
            # Load and augment image
            image = cv2.imread(source_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented_image = self.augment_image(image_rgb)
            
            # Save augmented image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            prefix = "made" if is_made_class else "missed"
            filename = f"{prefix}_aug_{timestamp}.jpg"
            filepath = os.path.join(class_path, filename)
            
            cv2.imwrite(filepath, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            
            # Update metadata
            if is_made_class:
                self.metadata["made_count"] += 1
            else:
                self.metadata["missed_count"] += 1
            self.metadata["total_images"] += 1
        
        self.save_metadata()
    
    def prepare_training_data(self):
        """Prepare data for training"""
        images = []
        labels = []
        
        # Load made images (label 1)
        for filename in os.listdir(self.made_path):
            if filename.endswith('.jpg'):
                filepath = os.path.join(self.made_path, filename)
                image = cv2.imread(filepath)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image_rgb)
                labels.append(1)
        
        # Load missed images (label 0)
        for filename in os.listdir(self.missed_path):
            if filename.endswith('.jpg'):
                filepath = os.path.join(self.missed_path, filename)
                image = cv2.imread(filepath)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image_rgb)
                labels.append(0)
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
        )
        
        return X_train, X_val, y_train, y_val
    
    def clear_dataset(self):
        """Clear all images from dataset"""
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)
        self.create_dataset_structure()
        self.metadata = {
            "created": datetime.now().isoformat(),
            "made_count": 0,
            "missed_count": 0,
            "total_images": 0,
            "last_updated": datetime.now().isoformat()
        }
        self.save_metadata()
    
    def export_dataset(self, export_path):
        """Export dataset to specified path"""
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        shutil.copytree(self.dataset_path, export_path)
        return export_path
