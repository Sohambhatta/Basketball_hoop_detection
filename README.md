# Basketball Hoop Detection Training System

A computer vision system for training a basketball hoop detection model using Jetson libraries with a PyQt5 GUI interface.

## Features

- Real-time camera feed for training data collection
- Binary classification: Ball in hoop vs Ball missed
- Confidence threshold optimization (>60%)
- Automatic epoch and dataset size optimization
- Live scoring counter (makes vs misses)
- Timer functionality
- Model testing and validation

## Requirements

- NVIDIA Jetson device (Nano, Xavier, Orin)
- USB/CSI camera
- Python 3.6+
- JetPack SDK installed

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training GUI:
```bash
python main_gui.py
```

## Usage

1. **Training Mode**:
   - Connect camera and start live feed
   - Capture images when ball is in hoop (label as "Made")
   - Capture images when ball misses (label as "Missed")
   - Set confidence threshold and optimization parameters
   - Train the model

2. **Testing Mode**:
   - Use trained model for real-time detection
   - Track makes and misses automatically
   - Monitor confidence scores and timing

## Project Structure

```
Basketball_hoop_detection/
├── main_gui.py              # Main GUI application
├── camera_handler.py        # Camera management
├── model_trainer.py         # Training logic
├── model_inference.py       # Inference and detection
├── data_manager.py          # Dataset management
├── config.py               # Configuration settings
├── utils/                  # Utility functions
├── models/                 # Trained models
├── datasets/              # Training datasets
└── logs/                  # Training logs
```
