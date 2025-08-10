"""
Configuration settings for Basketball Hoop Detection System
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_INDEX = 0  # Change if using different camera

# Training settings
MIN_CONFIDENCE_THRESHOLD = 0.6  # 60% minimum confidence
DEFAULT_EPOCHS = 50
MIN_IMAGES_PER_CLASS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Model settings
MODEL_INPUT_SIZE = (224, 224)
NUM_CLASSES = 2  # Made vs Missed

# GUI settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
VIDEO_DISPLAY_WIDTH = 640
VIDEO_DISPLAY_HEIGHT = 480

# Data augmentation settings
ROTATION_RANGE = 15
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = (0.8, 1.2)

# Jetson optimization settings
USE_TENSORRT = True
TENSORRT_PRECISION = 'FP16'  # FP32, FP16, INT8
