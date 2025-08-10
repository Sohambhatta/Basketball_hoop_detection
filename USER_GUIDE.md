# Basketball Hoop Detection Training System - User Guide

## Overview

This is a comprehensive computer vision system for training and deploying a basketball hoop detection model using NVIDIA Jetson libraries and PyQt5 GUI. The system can detect whether a basketball shot was made or missed with configurable confidence thresholds.

## Features

### Training Mode
- Real-time camera feed for data collection
- Manual labeling of shots (Made/Missed)
- Dataset management and statistics
- Automatic data augmentation
- Parameter optimization for target confidence (>60%)
- Progress tracking during training

### Testing/Live Mode
- Real-time inference using trained models
- Confidence threshold validation
- Shot counting and statistics
- Timer functionality
- Performance metrics

### GUI Features
- Tabbed interface for different functions
- Real-time camera display
- Training progress visualization
- Dataset statistics and management
- Comprehensive logging system
- Export capabilities for datasets and statistics

## Installation

### Prerequisites
1. NVIDIA Jetson device (Nano, Xavier, or Orin) - recommended
2. Python 3.6 or higher
3. USB or CSI camera
4. JetPack SDK (for Jetson devices)

### Setup Steps

1. **Clone or copy the project files to your Jetson device**

2. **Run the setup script:**
   ```bash
   python setup.py
   ```

3. **Install dependencies manually if setup fails:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   python launch.py
   ```
   or directly:
   ```bash
   python main_gui.py
   ```

## Usage Guide

### 1. Initial Setup

1. Connect your camera (USB or CSI)
2. Launch the application
3. Click "Start Camera" to begin camera feed
4. Verify the camera is working correctly

### 2. Training Data Collection

1. Select "Training Mode"
2. Position your basketball setup in view
3. Take shots at the hoop
4. For each shot, click either:
   - "Shot MADE ✓" (green button) - if ball went in
   - "Shot MISSED ✗" (red button) - if ball missed

**Tips for good training data:**
- Collect at least 100 images of each type (made/missed)
- Vary lighting conditions
- Include different ball positions and angles
- Maintain consistent camera position

### 3. Dataset Management

Navigate to the "Dataset" tab to:
- View current dataset statistics
- Check class balance (should be roughly equal)
- Augment dataset automatically
- Export dataset for backup
- Clear dataset if needed

### 4. Model Training

1. Go to "Training" tab to configure parameters:
   - **Epochs**: Number of training iterations (30-100)
   - **Learning Rate**: Training speed (0.001 recommended)
   - **Batch Size**: Images per training batch (16 recommended)
   - **Min Confidence**: Target confidence threshold (0.60+ required)

2. Enable "Auto-optimize parameters" for automatic tuning

3. Click "Start Training" and wait for completion

**Training will automatically:**
- Optimize epochs and parameters to meet confidence threshold
- Balance dataset through augmentation
- Validate performance on test data
- Save the best performing model

### 5. Testing and Live Detection

1. Switch to "Testing Mode" or "Live Mode"
2. Click "Load Latest Model" to load trained model
3. Click "Start Detection" for real-time inference
4. Monitor statistics in the "Statistics" tab

## Configuration

### Camera Settings
Edit `config.py` to modify camera parameters:
- `CAMERA_WIDTH`, `CAMERA_HEIGHT`: Resolution
- `CAMERA_FPS`: Frame rate
- `CAMERA_INDEX`: Camera device index

### Training Settings
- `MIN_CONFIDENCE_THRESHOLD`: Minimum required confidence (0.6)
- `DEFAULT_EPOCHS`: Default training epochs (50)
- `BATCH_SIZE`: Training batch size (16)
- `LEARNING_RATE`: Learning rate (0.001)

### Model Settings
- `MODEL_INPUT_SIZE`: Input image size (224x224)
- `NUM_CLASSES`: Number of classes (2 - made/missed)

## Troubleshooting

### Camera Issues
- **"Camera not found"**: Check camera connection and permissions
- **Poor image quality**: Adjust lighting and camera position
- **Slow frame rate**: Reduce resolution in config.py

### Training Issues
- **"Insufficient data"**: Collect at least 20+ images per class
- **Low accuracy**: Ensure good quality, balanced training data
- **Training fails**: Check GPU memory and reduce batch size

### Model Issues
- **"Model not found"**: Complete training first or check models/ directory
- **Low confidence predictions**: Retrain with more/better data
- **Slow inference**: Enable TensorRT optimization in config

### Environment Issues
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **PyQt5 errors**: Install system Qt libraries
- **CUDA not available**: Check GPU drivers and CUDA installation

## Performance Optimization

### For Jetson Devices
1. Enable TensorRT optimization in config.py
2. Use FP16 precision for better speed
3. Optimize camera pipeline for CSI cameras
4. Monitor GPU/CPU usage during training

### Data Quality Tips
1. Maintain consistent lighting
2. Keep camera position stable
3. Include edge cases in training data
4. Balance dataset classes equally
5. Use data augmentation sparingly

## File Structure

```
Basketball_hoop_detection/
├── main_gui.py              # Main GUI application
├── launch.py               # Application launcher
├── setup.py                # Environment setup
├── config.py               # Configuration settings
├── camera_handler.py       # Camera management
├── data_manager.py         # Dataset handling
├── model_trainer.py        # Training logic
├── model_inference.py      # Inference engine
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── USER_GUIDE.md          # This file
├── utils/
│   ├── __init__.py        # Utils package
│   └── helpers.py         # Helper functions
├── models/                # Saved models
├── datasets/              # Training data
└── logs/                  # Training logs
```

## Advanced Usage

### Custom Model Architecture
Modify `BasketballNet` class in `model_trainer.py` to use different architectures.

### Custom Data Augmentation
Edit augmentation parameters in `data_manager.py` and `config.py`.

### Batch Processing
Use `model_inference.py` directly for batch processing of images.

### API Integration
Import modules directly for programmatic use without GUI.

## Support and Contributions

This system is designed for basketball shot detection but can be adapted for other binary classification computer vision tasks by:
1. Modifying class labels in the code
2. Adjusting model architecture if needed
3. Updating GUI text and labels
4. Retraining with appropriate data

## Performance Expectations

### Typical Results
- **Training Time**: 10-30 minutes (depending on dataset size)
- **Accuracy**: 85-95% with good training data
- **Inference Speed**: 10-30 FPS on Jetson devices
- **Memory Usage**: ~2GB GPU memory during training

### Best Practices
1. Start with small dataset for testing
2. Gradually increase data quality and quantity
3. Monitor validation accuracy during training
4. Test model thoroughly before deployment
5. Retrain periodically with new data

---

For technical support or questions, refer to the logs tab in the GUI or check the console output for detailed error messages.
