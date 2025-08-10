"""
Utility functions for Basketball Hoop Detection System
"""

import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import json

def create_directories():
    """Create necessary project directories"""
    from config import MODELS_DIR, DATASETS_DIR, LOGS_DIR
    
    directories = [MODELS_DIR, DATASETS_DIR, LOGS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def format_time(seconds):
    """Format seconds into readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(remaining_minutes)}m"

def resize_image_keep_aspect(image, target_size):
    """Resize image while keeping aspect ratio"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create blank canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate position to center image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def draw_prediction_overlay(image, prediction_result, show_confidence=True):
    """Draw prediction overlay on image"""
    overlay = image.copy()
    h, w = image.shape[:2]
    
    # Define colors
    made_color = (0, 255, 0)  # Green
    missed_color = (0, 0, 255)  # Red
    low_confidence_color = (0, 255, 255)  # Yellow
    
    if prediction_result is None:
        return overlay
    
    # Determine color and text
    if prediction_result['meets_threshold']:
        color = made_color if prediction_result['is_made'] else missed_color
        status_text = "MADE" if prediction_result['is_made'] else "MISSED"
    else:
        color = low_confidence_color
        status_text = "LOW CONFIDENCE"
    
    # Draw status text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(w, h) / 400  # Scale font based on image size
    thickness = max(1, int(font_scale * 2))
    
    # Status text
    (text_w, text_h), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)
    cv2.rectangle(overlay, (10, 10), (10 + text_w + 20, 10 + text_h + baseline + 20), color, -1)
    cv2.putText(overlay, status_text, (20, 10 + text_h + 10), font, font_scale, (255, 255, 255), thickness)
    
    # Confidence text
    if show_confidence:
        confidence_text = f"Confidence: {prediction_result['confidence']:.2f}"
        (conf_w, conf_h), baseline = cv2.getTextSize(confidence_text, font, font_scale * 0.7, thickness)
        y_pos = 10 + text_h + baseline + 40
        cv2.rectangle(overlay, (10, y_pos), (10 + conf_w + 20, y_pos + conf_h + baseline + 20), (50, 50, 50), -1)
        cv2.putText(overlay, confidence_text, (20, y_pos + conf_h + 10), font, font_scale * 0.7, (255, 255, 255), thickness)
    
    return overlay

def draw_statistics_overlay(image, stats, position='top_right'):
    """Draw statistics overlay on image"""
    overlay = image.copy()
    h, w = image.shape[:2]
    
    # Prepare statistics text
    stats_lines = [
        f"Total: {stats['total_predictions']}",
        f"Made: {stats['made_predictions']} ({stats['made_percentage']:.1f}%)",
        f"Missed: {stats['missed_predictions']} ({stats['missed_percentage']:.1f}%)",
        f"High Conf: {stats['high_confidence_predictions']} ({stats['high_confidence_percentage']:.1f}%)"
    ]
    
    if 'elapsed_time' in stats:
        stats_lines.append(f"Time: {format_time(stats['elapsed_time'])}")
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(w, h) / 800
    thickness = max(1, int(font_scale * 2))
    line_spacing = int(25 * font_scale)
    
    # Calculate text dimensions
    max_width = 0
    total_height = 0
    text_sizes = []
    
    for line in stats_lines:
        (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        text_sizes.append((text_w, text_h, baseline))
        max_width = max(max_width, text_w)
        total_height += text_h + line_spacing
    
    # Determine position
    padding = 10
    if position == 'top_right':
        start_x = w - max_width - padding * 3
        start_y = padding * 2
    elif position == 'top_left':
        start_x = padding
        start_y = padding * 2
    elif position == 'bottom_right':
        start_x = w - max_width - padding * 3
        start_y = h - total_height - padding * 2
    else:  # bottom_left
        start_x = padding
        start_y = h - total_height - padding * 2
    
    # Draw background rectangle
    cv2.rectangle(overlay, 
                 (start_x - padding, start_y - padding),
                 (start_x + max_width + padding * 2, start_y + total_height + padding),
                 (0, 0, 0, 180), -1)
    
    # Draw text lines
    current_y = start_y
    for i, line in enumerate(stats_lines):
        text_w, text_h, baseline = text_sizes[i]
        cv2.putText(overlay, line, (start_x, current_y + text_h), 
                   font, font_scale, (255, 255, 255), thickness)
        current_y += text_h + line_spacing
    
    return overlay

def save_prediction_image(image, prediction_result, save_dir="predictions"):
    """Save image with prediction overlay"""
    from config import PROJECT_ROOT
    
    save_path = os.path.join(PROJECT_ROOT, save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    # Add prediction overlay
    overlay_image = draw_prediction_overlay(image, prediction_result)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    status = "made" if prediction_result['is_made'] else "missed"
    confidence = prediction_result['confidence']
    filename = f"{status}_{confidence:.3f}_{timestamp}.jpg"
    
    filepath = os.path.join(save_path, filename)
    cv2.imwrite(filepath, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
    
    return filepath

def calculate_dataset_balance(data_manager):
    """Calculate dataset balance recommendations"""
    stats = data_manager.get_dataset_stats()
    
    made_count = stats['made_count']
    missed_count = stats['missed_count']
    total_count = stats['total_images']
    
    if total_count == 0:
        return {
            'is_balanced': False,
            'recommendation': "Start collecting training data",
            'target_made': 100,
            'target_missed': 100,
            'current_balance': 0.0
        }
    
    # Calculate balance ratio (ideally should be close to 1.0)
    balance_ratio = made_count / max(missed_count, 1)
    
    # Determine if balanced (within 20% of each other)
    is_balanced = 0.8 <= balance_ratio <= 1.25
    
    # Generate recommendations
    min_per_class = 100  # Minimum recommended images per class
    
    if total_count < min_per_class * 2:
        if made_count < min_per_class:
            need_made = min_per_class - made_count
        else:
            need_made = 0
            
        if missed_count < min_per_class:
            need_missed = min_per_class - missed_count
        else:
            need_missed = 0
        
        recommendation = f"Need more data: {need_made} more 'made' shots, {need_missed} more 'missed' shots"
    elif not is_balanced:
        if balance_ratio < 0.8:
            need_made = int(missed_count * 0.9) - made_count
            recommendation = f"Need {need_made} more 'made' shots to balance dataset"
        else:
            need_missed = int(made_count * 0.9) - missed_count
            recommendation = f"Need {need_missed} more 'missed' shots to balance dataset"
    else:
        recommendation = "Dataset is well balanced"
    
    return {
        'is_balanced': is_balanced,
        'recommendation': recommendation,
        'target_made': max(min_per_class, made_count),
        'target_missed': max(min_per_class, missed_count),
        'current_balance': balance_ratio,
        'total_images': total_count
    }

def validate_jetson_environment():
    """Validate Jetson environment and dependencies"""
    issues = []
    
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available - GPU acceleration disabled")
    except ImportError:
        issues.append("PyTorch not installed")
    
    # Check Jetson inference
    try:
        import jetson.inference
        import jetson.utils
    except ImportError:
        issues.append("Jetson Inference not available - using OpenCV camera fallback")
    
    # Check OpenCV
    try:
        import cv2
        if not cv2.getBuildInformation():
            issues.append("OpenCV installation issue detected")
    except ImportError:
        issues.append("OpenCV not available")
    
    # Check PyQt5
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        issues.append("PyQt5 not available - GUI will not work")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'recommendations': _get_environment_recommendations(issues)
    }

def _get_environment_recommendations(issues):
    """Get recommendations for environment issues"""
    recommendations = []
    
    for issue in issues:
        if "PyTorch" in issue:
            recommendations.append("Install PyTorch: pip install torch torchvision")
        elif "Jetson Inference" in issue:
            recommendations.append("Install Jetson Inference following NVIDIA's guide")
        elif "OpenCV" in issue:
            recommendations.append("Install OpenCV: pip install opencv-python")
        elif "PyQt5" in issue:
            recommendations.append("Install PyQt5: pip install PyQt5")
        elif "CUDA" in issue:
            recommendations.append("Check CUDA installation and GPU drivers")
    
    return recommendations

class Timer:
    """Simple timer utility"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.is_running = False
    
    def start(self):
        """Start the timer"""
        self.start_time = datetime.now()
        self.end_time = None
        self.is_running = True
    
    def stop(self):
        """Stop the timer"""
        if self.is_running:
            self.end_time = datetime.now()
            self.is_running = False
    
    def reset(self):
        """Reset the timer"""
        self.start_time = None
        self.end_time = None
        self.is_running = False
    
    def elapsed(self):
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else datetime.now()
        delta = end - self.start_time
        return delta.total_seconds()
    
    def elapsed_formatted(self):
        """Get formatted elapsed time string"""
        return format_time(self.elapsed())
