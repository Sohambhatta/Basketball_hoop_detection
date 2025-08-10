"""
Camera Handler for Basketball Hoop Detection System
Handles camera initialization, frame capture, and Jetson optimization
"""

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from config import *

# Optional Jetson imports
try:
    import jetson.utils
    import jetson.inference
    JETSON_AVAILABLE = True
except ImportError:
    print("Warning: Jetson modules not available - using OpenCV fallback")
    jetson = None
    JETSON_AVAILABLE = False

class CameraThread(QThread):
    """Thread for handling camera operations"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.running = False
        self.capture_next_frame = False
        self.captured_frames = []
        
    def initialize_camera(self):
        """Initialize camera with Jetson optimizations"""
        try:
            # Try to use Jetson camera first (CSI camera) if available
            if JETSON_AVAILABLE:
                self.camera = jetson.utils.videoSource("csi://0")
                print("Initialized CSI camera")
            else:
                raise ImportError("Jetson not available")
        except:
            try:
                # Fallback to USB camera
                self.camera = cv2.VideoCapture(CAMERA_INDEX)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                print("Initialized USB camera")
            except Exception as e:
                print(f"Failed to initialize camera: {e}")
                return False
        return True
    
    def run(self):
        """Main camera loop"""
        if not self.initialize_camera():
            return
            
        self.running = True
        while self.running:
            try:
                if isinstance(self.camera, cv2.VideoCapture):
                    ret, frame = self.camera.read()
                    if ret:
                        # Convert BGR to RGB for Qt display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_ready.emit(frame_rgb)
                        
                        # Capture frame if requested
                        if self.capture_next_frame:
                            self.captured_frames.append(frame_rgb.copy())
                            self.capture_next_frame = False
                else:
                    # Jetson camera
                    if JETSON_AVAILABLE:
                        frame = self.camera.Capture()
                        if frame is not None:
                            # Convert CUDA to numpy
                            frame_np = jetson.utils.cudaToNumpy(frame)
                            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2RGB)
                            self.frame_ready.emit(frame_rgb)
                            
                            # Capture frame if requested
                            if self.capture_next_frame:
                                self.captured_frames.append(frame_rgb.copy())
                                self.capture_next_frame = False
                            
                self.msleep(33)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera error: {e}")
                break
    
    def capture_frame(self):
        """Request to capture the next frame"""
        self.capture_next_frame = True
    
    def get_captured_frames(self):
        """Get and clear captured frames"""
        frames = self.captured_frames.copy()
        self.captured_frames.clear()
        return frames
    
    def stop(self):
        """Stop camera thread"""
        self.running = False
        if self.camera:
            if isinstance(self.camera, cv2.VideoCapture):
                self.camera.release()
            else:
                del self.camera
        self.wait()

class CameraHandler:
    """Main camera handler class"""
    
    def __init__(self):
        self.camera_thread = CameraThread()
        self.is_recording = False
        
    def start_camera(self):
        """Start camera thread"""
        self.camera_thread.start()
        
    def stop_camera(self):
        """Stop camera thread"""
        self.camera_thread.stop()
        
    def capture_image(self):
        """Capture current frame"""
        self.camera_thread.capture_frame()
        
    def get_captured_images(self):
        """Get captured frames"""
        return self.camera_thread.get_captured_frames()
    
    def connect_frame_ready(self, slot):
        """Connect frame ready signal to slot"""
        self.camera_thread.frame_ready.connect(slot)
    
    @staticmethod
    def numpy_to_qimage(np_img):
        """Convert numpy array to QImage"""
        h, w, ch = np_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(np_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qt_image
    
    @staticmethod
    def qimage_to_pixmap(qt_image):
        """Convert QImage to QPixmap"""
        return QPixmap.fromImage(qt_image)
