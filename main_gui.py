"""
Main GUI Application for Basketball Hoop Detection Training System
PyQt5-based interface for training, testing, and real-time detection
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                            QTextEdit, QSlider, QSpinBox, QDoubleSpinBox,
                            QProgressBar, QGroupBox, QTabWidget, QFrame,
                            QCheckBox, QComboBox, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPixmap, QPalette

# Import project modules
from camera_handler import CameraHandler
from data_manager import DataManager
from model_trainer import ModelTrainer
from model_inference import ModelInference
from utils.helpers import Timer, format_time, validate_jetson_environment
from config import *

class BasketballDetectionGUI(QMainWindow):
    """Main GUI application for basketball hoop detection"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basketball Hoop Detection Training System")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Initialize components
        self.camera_handler = CameraHandler()
        self.data_manager = DataManager()
        self.model_trainer = None
        self.model_inference = ModelInference()
        self.timer = Timer()
        
        # State variables
        self.current_mode = "training"  # training, testing, live
        self.is_camera_running = False
        self.auto_capture_enabled = False
        self.training_in_progress = False
        
        # Statistics
        self.session_stats = {
            'made_count': 0,
            'missed_count': 0,
            'total_shots': 0,
            'session_start': None
        }
        
        self.init_ui()
        self.setup_connections()
        self.update_display_timer = QTimer()
        self.update_display_timer.timeout.connect(self.update_displays)
        self.update_display_timer.start(100)  # Update every 100ms
        
        # Validate environment
        self.validate_environment()
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Camera and controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 2)
        
        # Right panel - Settings and stats
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
    
    def create_left_panel(self):
        """Create left panel with camera feed and main controls"""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(VIDEO_DISPLAY_WIDTH, VIDEO_DISPLAY_HEIGHT)
        self.camera_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera Feed")
        layout.addWidget(self.camera_label)
        
        # Mode selection
        mode_group = QGroupBox("Mode Selection")
        mode_layout = QHBoxLayout(mode_group)
        
        self.training_mode_btn = QPushButton("Training Mode")
        self.training_mode_btn.setCheckable(True)
        self.training_mode_btn.setChecked(True)
        self.training_mode_btn.clicked.connect(lambda: self.set_mode("training"))
        
        self.testing_mode_btn = QPushButton("Testing Mode")
        self.testing_mode_btn.setCheckable(True)
        self.testing_mode_btn.clicked.connect(lambda: self.set_mode("testing"))
        
        self.live_mode_btn = QPushButton("Live Mode")
        self.live_mode_btn.setCheckable(True)
        self.live_mode_btn.clicked.connect(lambda: self.set_mode("live"))
        
        mode_layout.addWidget(self.training_mode_btn)
        mode_layout.addWidget(self.testing_mode_btn)
        mode_layout.addWidget(self.live_mode_btn)
        layout.addWidget(mode_group)
        
        # Camera controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QHBoxLayout(camera_group)
        
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        
        self.capture_btn = QPushButton("Capture Frame")
        self.capture_btn.clicked.connect(self.capture_frame)
        self.capture_btn.setEnabled(False)
        
        camera_layout.addWidget(self.start_camera_btn)
        camera_layout.addWidget(self.capture_btn)
        layout.addWidget(camera_group)
        
        # Training controls
        self.training_group = QGroupBox("Training Controls")
        training_layout = QVBoxLayout(self.training_group)
        
        # Labeling buttons
        label_layout = QHBoxLayout()
        self.made_shot_btn = QPushButton("Shot MADE ✓")
        self.made_shot_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.made_shot_btn.clicked.connect(lambda: self.label_and_capture(True))
        self.made_shot_btn.setEnabled(False)
        
        self.missed_shot_btn = QPushButton("Shot MISSED ✗")
        self.missed_shot_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.missed_shot_btn.clicked.connect(lambda: self.label_and_capture(False))
        self.missed_shot_btn.setEnabled(False)
        
        label_layout.addWidget(self.made_shot_btn)
        label_layout.addWidget(self.missed_shot_btn)
        training_layout.addLayout(label_layout)
        
        # Training progress
        self.train_model_btn = QPushButton("Start Training")
        self.train_model_btn.clicked.connect(self.start_training)
        
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        
        training_layout.addWidget(self.train_model_btn)
        training_layout.addWidget(self.training_progress)
        
        layout.addWidget(self.training_group)
        
        # Testing/Live controls
        self.detection_group = QGroupBox("Detection Controls")
        detection_layout = QVBoxLayout(self.detection_group)
        
        self.load_model_btn = QPushButton("Load Latest Model")
        self.load_model_btn.clicked.connect(self.load_model)
        
        self.start_detection_btn = QPushButton("Start Detection")
        self.start_detection_btn.clicked.connect(self.toggle_detection)
        self.start_detection_btn.setEnabled(False)
        
        detection_layout.addWidget(self.load_model_btn)
        detection_layout.addWidget(self.start_detection_btn)
        
        self.detection_group.setVisible(False)
        layout.addWidget(self.detection_group)
        
        return left_widget
    
    def create_right_panel(self):
        """Create right panel with settings and statistics"""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        # Tab widget for different panels
        tab_widget = QTabWidget()
        
        # Dataset tab
        dataset_tab = self.create_dataset_tab()
        tab_widget.addTab(dataset_tab, "Dataset")
        
        # Training tab
        training_tab = self.create_training_tab()
        tab_widget.addTab(training_tab, "Training")
        
        # Statistics tab
        stats_tab = self.create_statistics_tab()
        tab_widget.addTab(stats_tab, "Statistics")
        
        # Logs tab
        logs_tab = self.create_logs_tab()
        tab_widget.addTab(logs_tab, "Logs")
        
        layout.addWidget(tab_widget)
        
        return right_widget
    
    def create_dataset_tab(self):
        """Create dataset information tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Dataset stats
        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.made_count_label = QLabel("Made shots: 0")
        self.missed_count_label = QLabel("Missed shots: 0")
        self.total_images_label = QLabel("Total images: 0")
        self.balance_label = QLabel("Balance: N/A")
        
        stats_layout.addWidget(self.made_count_label, 0, 0)
        stats_layout.addWidget(self.missed_count_label, 0, 1)
        stats_layout.addWidget(self.total_images_label, 1, 0)
        stats_layout.addWidget(self.balance_label, 1, 1)
        
        layout.addWidget(stats_group)
        
        # Dataset actions
        actions_group = QGroupBox("Dataset Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.augment_btn = QPushButton("Augment Dataset")
        self.augment_btn.clicked.connect(self.augment_dataset)
        
        self.clear_dataset_btn = QPushButton("Clear Dataset")
        self.clear_dataset_btn.clicked.connect(self.clear_dataset)
        
        self.export_dataset_btn = QPushButton("Export Dataset")
        self.export_dataset_btn.clicked.connect(self.export_dataset)
        
        actions_layout.addWidget(self.augment_btn)
        actions_layout.addWidget(self.clear_dataset_btn)
        actions_layout.addWidget(self.export_dataset_btn)
        
        layout.addWidget(actions_group)
        
        layout.addStretch()
        
        return widget
    
    def create_training_tab(self):
        """Create training parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_group)
        
        # Epochs
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(10, 200)
        self.epochs_spinbox.setValue(DEFAULT_EPOCHS)
        params_layout.addWidget(self.epochs_spinbox, 0, 1)
        
        # Learning rate
        params_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.00001, 0.1)
        self.lr_spinbox.setValue(LEARNING_RATE)
        self.lr_spinbox.setDecimals(5)
        self.lr_spinbox.setSingleStep(0.0001)
        params_layout.addWidget(self.lr_spinbox, 1, 1)
        
        # Batch size
        params_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(4, 64)
        self.batch_size_spinbox.setValue(BATCH_SIZE)
        params_layout.addWidget(self.batch_size_spinbox, 2, 1)
        
        # Confidence threshold
        params_layout.addWidget(QLabel("Min Confidence:"), 3, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(int(MIN_CONFIDENCE_THRESHOLD * 100))
        self.confidence_value_label = QLabel(f"{MIN_CONFIDENCE_THRESHOLD:.2f}")
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_value_label)
        params_layout.addLayout(confidence_layout, 3, 1)
        
        layout.addWidget(params_group)
        
        # Auto-optimization
        auto_group = QGroupBox("Auto-Optimization")
        auto_layout = QVBoxLayout(auto_group)
        
        self.auto_optimize_checkbox = QCheckBox("Auto-optimize parameters")
        self.auto_optimize_checkbox.setChecked(True)
        auto_layout.addWidget(self.auto_optimize_checkbox)
        
        layout.addWidget(auto_group)
        
        layout.addStretch()
        
        return widget
    
    def create_statistics_tab(self):
        """Create statistics display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Session stats
        session_group = QGroupBox("Session Statistics")
        session_layout = QGridLayout(session_group)
        
        self.session_made_label = QLabel("Made: 0")
        self.session_missed_label = QLabel("Missed: 0")
        self.session_total_label = QLabel("Total: 0")
        self.session_accuracy_label = QLabel("Accuracy: N/A")
        self.session_time_label = QLabel("Time: 00:00")
        
        session_layout.addWidget(self.session_made_label, 0, 0)
        session_layout.addWidget(self.session_missed_label, 0, 1)
        session_layout.addWidget(self.session_total_label, 1, 0)
        session_layout.addWidget(self.session_accuracy_label, 1, 1)
        session_layout.addWidget(self.session_time_label, 2, 0, 1, 2)
        
        layout.addWidget(session_group)
        
        # Model info
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_group)
        
        self.model_info_label = QLabel("No model loaded")
        model_layout.addWidget(self.model_info_label)
        
        layout.addWidget(model_group)
        
        # Timer controls
        timer_group = QGroupBox("Timer Controls")
        timer_layout = QHBoxLayout(timer_group)
        
        self.start_timer_btn = QPushButton("Start Timer")
        self.start_timer_btn.clicked.connect(self.toggle_timer)
        
        self.reset_stats_btn = QPushButton("Reset Statistics")
        self.reset_stats_btn.clicked.connect(self.reset_statistics)
        
        timer_layout.addWidget(self.start_timer_btn)
        timer_layout.addWidget(self.reset_stats_btn)
        
        layout.addWidget(timer_group)
        
        layout.addStretch()
        
        return widget
    
    def create_logs_tab(self):
        """Create logs display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_display)
        
        # Log controls
        log_controls = QHBoxLayout()
        
        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.clicked.connect(self.clear_logs)
        
        self.save_logs_btn = QPushButton("Save Logs")
        self.save_logs_btn.clicked.connect(self.save_logs)
        
        log_controls.addWidget(self.clear_logs_btn)
        log_controls.addWidget(self.save_logs_btn)
        log_controls.addStretch()
        
        layout.addLayout(log_controls)
        
        return widget
    
    def setup_connections(self):
        """Setup signal connections"""
        # Camera connections
        self.camera_handler.connect_frame_ready(self.update_camera_display)
    
    def validate_environment(self):
        """Validate the environment setup"""
        validation = validate_jetson_environment()
        
        if not validation['is_valid']:
            self.log_message("Environment validation issues found:")
            for issue in validation['issues']:
                self.log_message(f"  - {issue}")
            
            self.log_message("Recommendations:")
            for rec in validation['recommendations']:
                self.log_message(f"  - {rec}")
        else:
            self.log_message("Environment validation successful!")
    
    def set_mode(self, mode):
        """Set the current operating mode"""
        self.current_mode = mode
        
        # Update button states
        self.training_mode_btn.setChecked(mode == "training")
        self.testing_mode_btn.setChecked(mode == "testing")
        self.live_mode_btn.setChecked(mode == "live")
        
        # Show/hide relevant groups
        self.training_group.setVisible(mode == "training")
        self.detection_group.setVisible(mode in ["testing", "live"])
        
        self.log_message(f"Switched to {mode} mode")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.is_camera_running:
            self.camera_handler.start_camera()
            self.is_camera_running = True
            self.start_camera_btn.setText("Stop Camera")
            self.capture_btn.setEnabled(True)
            self.made_shot_btn.setEnabled(True)
            self.missed_shot_btn.setEnabled(True)
            self.log_message("Camera started")
        else:
            self.camera_handler.stop_camera()
            self.is_camera_running = False
            self.start_camera_btn.setText("Start Camera")
            self.capture_btn.setEnabled(False)
            self.made_shot_btn.setEnabled(False)
            self.missed_shot_btn.setEnabled(False)
            self.camera_label.setText("Camera Feed")
            self.log_message("Camera stopped")
    
    def update_camera_display(self, frame):
        """Update camera display with new frame"""
        try:
            # Convert numpy array to QImage
            qt_image = self.camera_handler.numpy_to_qimage(frame)
            
            # Scale image to fit display
            scaled_image = qt_image.scaled(
                self.camera_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(scaled_image)
            self.camera_label.setPixmap(pixmap)
            
        except Exception as e:
            self.log_message(f"Error updating camera display: {e}")
    
    def capture_frame(self):
        """Capture current frame"""
        if self.is_camera_running:
            self.camera_handler.capture_image()
            self.log_message("Frame captured")
    
    def label_and_capture(self, is_made):
        """Label and capture current frame for training"""
        if not self.is_camera_running:
            self.log_message("Camera not running")
            return
        
        self.camera_handler.capture_image()
        
        # Get captured images
        captured_frames = self.camera_handler.get_captured_images()
        
        if captured_frames:
            for frame in captured_frames:
                self.data_manager.add_image(frame, is_made)
            
            label_text = "MADE" if is_made else "MISSED"
            self.log_message(f"Added {len(captured_frames)} image(s) labeled as {label_text}")
            self.update_dataset_stats()
    
    def start_training(self):
        """Start model training"""
        if self.training_in_progress:
            self.log_message("Training already in progress")
            return
        
        # Check dataset
        stats = self.data_manager.get_dataset_stats()
        if stats['total_images'] < 20:
            QMessageBox.warning(self, "Insufficient Data", 
                              "Need at least 20 images to start training")
            return
        
        # Get training parameters
        epochs = self.epochs_spinbox.value()
        learning_rate = self.lr_spinbox.value()
        batch_size = self.batch_size_spinbox.value()
        confidence_threshold = self.confidence_slider.value() / 100.0
        
        # Initialize trainer
        self.model_trainer = ModelTrainer(self.data_manager, confidence_threshold)
        self.model_trainer.set_training_params(epochs, learning_rate, batch_size)
        
        # Connect signals
        self.model_trainer.progress_updated.connect(self.update_training_progress)
        self.model_trainer.training_completed.connect(self.training_completed)
        self.model_trainer.log_message.connect(self.log_message)
        
        # Start training
        self.training_in_progress = True
        self.train_model_btn.setText("Stop Training")
        self.training_progress.setVisible(True)
        self.training_progress.setValue(0)
        
        self.model_trainer.start()
        self.log_message("Training started...")
    
    def update_training_progress(self, epoch, train_loss, val_accuracy):
        """Update training progress display"""
        max_epochs = self.epochs_spinbox.value()
        progress = int((epoch / max_epochs) * 100)
        self.training_progress.setValue(progress)
        
        self.log_message(f"Epoch {epoch}/{max_epochs}: Loss={train_loss:.4f}, Acc={val_accuracy:.4f}")
    
    def training_completed(self, model_path, final_accuracy):
        """Handle training completion"""
        self.training_in_progress = False
        self.train_model_btn.setText("Start Training")
        self.training_progress.setVisible(False)
        
        if model_path and final_accuracy >= self.confidence_slider.value() / 100.0:
            self.log_message(f"Training completed successfully! Final accuracy: {final_accuracy:.4f}")
            self.log_message(f"Model saved to: {model_path}")
            
            # Automatically load the new model if in testing/live mode
            if self.current_mode in ["testing", "live"]:
                self.load_model(model_path)
        else:
            self.log_message("Training completed but target confidence not reached")
    
    def load_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = self.model_inference.get_latest_model()
        
        if model_path and self.model_inference.load_model(model_path):
            model_info = self.model_inference.get_model_info()
            self.model_info_label.setText(
                f"Model loaded\nAccuracy: {model_info['accuracy']:.3f}\n"
                f"Architecture: {model_info['architecture']}"
            )
            self.start_detection_btn.setEnabled(True)
            self.log_message(f"Model loaded: {os.path.basename(model_path)}")
        else:
            self.log_message("Failed to load model")
    
    def toggle_detection(self):
        """Toggle real-time detection"""
        # This would be implemented with additional threading for real-time inference
        self.log_message("Detection toggle not yet implemented")
    
    def update_dataset_stats(self):
        """Update dataset statistics display"""
        stats = self.data_manager.get_dataset_stats()
        
        self.made_count_label.setText(f"Made shots: {stats['made_count']}")
        self.missed_count_label.setText(f"Missed shots: {stats['missed_count']}")
        self.total_images_label.setText(f"Total images: {stats['total_images']}")
        
        if stats['total_images'] > 0:
            balance_ratio = stats['made_count'] / max(stats['missed_count'], 1)
            self.balance_label.setText(f"Balance: {balance_ratio:.2f}")
        else:
            self.balance_label.setText("Balance: N/A")
    
    def update_confidence_label(self):
        """Update confidence threshold label"""
        value = self.confidence_slider.value() / 100.0
        self.confidence_value_label.setText(f"{value:.2f}")
    
    def update_displays(self):
        """Update all displays periodically"""
        self.update_dataset_stats()
        
        # Update timer display
        if self.timer.is_running:
            elapsed = self.timer.elapsed_formatted()
            self.session_time_label.setText(f"Time: {elapsed}")
    
    def toggle_timer(self):
        """Toggle session timer"""
        if not self.timer.is_running:
            self.timer.start()
            self.start_timer_btn.setText("Stop Timer")
            self.session_stats['session_start'] = datetime.now()
        else:
            self.timer.stop()
            self.start_timer_btn.setText("Start Timer")
    
    def reset_statistics(self):
        """Reset session statistics"""
        self.session_stats = {
            'made_count': 0,
            'missed_count': 0,
            'total_shots': 0,
            'session_start': None
        }
        
        self.session_made_label.setText("Made: 0")
        self.session_missed_label.setText("Missed: 0")
        self.session_total_label.setText("Total: 0")
        self.session_accuracy_label.setText("Accuracy: N/A")
        
        if self.model_inference.is_model_loaded():
            self.model_inference.reset_stats()
        
        self.timer.reset()
        self.start_timer_btn.setText("Start Timer")
        self.session_time_label.setText("Time: 00:00")
        
        self.log_message("Statistics reset")
    
    def augment_dataset(self):
        """Augment dataset to balance classes"""
        try:
            stats = self.data_manager.get_dataset_stats()
            target_per_class = max(stats['made_count'], stats['missed_count']) * 2
            
            self.log_message(f"Augmenting dataset to {target_per_class} images per class...")
            self.data_manager.augment_dataset(target_per_class)
            self.log_message("Dataset augmentation completed")
            
        except Exception as e:
            self.log_message(f"Error during augmentation: {e}")
    
    def clear_dataset(self):
        """Clear all dataset images"""
        reply = QMessageBox.question(
            self, 'Clear Dataset',
            'Are you sure you want to clear all training data?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.data_manager.clear_dataset()
            self.log_message("Dataset cleared")
    
    def export_dataset(self):
        """Export dataset to chosen location"""
        folder = QFileDialog.getExistingDirectory(self, "Select Export Location")
        if folder:
            export_path = os.path.join(folder, f"basketball_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.data_manager.export_dataset(export_path)
            self.log_message(f"Dataset exported to: {export_path}")
    
    def log_message(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_display.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_logs(self):
        """Clear log display"""
        self.log_display.clear()
    
    def save_logs(self):
        """Save logs to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Logs", 
            f"basketball_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_display.toPlainText())
            self.log_message(f"Logs saved to: {filename}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        if self.is_camera_running:
            self.camera_handler.stop_camera()
        
        if self.training_in_progress and self.model_trainer:
            self.model_trainer.stop_training()
        
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = BasketballDetectionGUI()
    window.show()
    
    # Start application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
