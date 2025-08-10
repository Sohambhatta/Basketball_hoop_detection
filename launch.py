"""
Basketball Hoop Detection System Launcher
This script handles environment validation and launches the GUI
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        missing_deps.append("PyQt5")
    
    # Optional dependencies (Jetson-specific)
    jetson_available = True
    try:
        import torch
    except ImportError:
        print("Warning: PyTorch not available - training will be limited")
        jetson_available = False
    
    try:
        import jetson.inference
        import jetson.utils
    except ImportError:
        print("Warning: Jetson inference not available - using OpenCV fallback")
        jetson_available = False
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print(f"\nInstall with: pip install {' '.join(missing_deps)}")
        return False
    
    return True

def launch_gui():
    """Launch the main GUI application"""
    try:
        # Import and run the main application
        from main_gui import main
        main()
    except ImportError as e:
        print(f"Error importing main GUI: {e}")
        print("Make sure all files are in place and dependencies are installed")
        return False
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return False

def main():
    """Main launcher function"""
    print("Basketball Hoop Detection System")
    print("=" * 35)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again")
        return
    
    print("✓ Dependencies OK")
    print("Launching GUI...")
    
    # Launch the application
    launch_gui()

if __name__ == "__main__":
    main()
