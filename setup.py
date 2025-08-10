"""
Setup script for Basketball Hoop Detection System
Run this to set up the environment and install dependencies
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ['models', 'datasets', 'logs', 'utils']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Directories created")

def check_jetson_environment():
    """Check if running on Jetson device"""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            content = f.read()
        if 'tegra' in content.lower():
            print("✓ Jetson environment detected")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠ Not running on Jetson device - some features may be limited")
    return False

def main():
    """Main setup function"""
    print("Basketball Hoop Detection System Setup")
    print("=" * 40)
    
    # Setup directories
    setup_directories()
    
    # Check environment
    is_jetson = check_jetson_environment()
    
    # Install requirements
    if install_requirements():
        print("\n✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Connect your camera")
        print("2. Run: python main_gui.py")
        print("3. Start collecting training data")
    else:
        print("\n✗ Setup failed. Please install dependencies manually.")
        
    if not is_jetson:
        print("\nNote: For best performance, run this on an NVIDIA Jetson device")

if __name__ == "__main__":
    main()
