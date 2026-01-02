#!/usr/bin/env python3
"""
Jetson Nano System Diagnostics for Human Detection
Helps diagnose "Illegal instruction" errors and other compatibility issues.
"""

import cv2
import numpy as np
import sys
import os
import subprocess
import platform


def check_system_info():
    """Check basic system information"""
    print("=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check if CUDA is available
    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA devices available: {cuda_devices}")
    except:
        print("CUDA not available or not compiled with OpenCV")
    
    # Check DNN module
    try:
        print(f"DNN module available: {hasattr(cv2, 'dnn')}")
        if hasattr(cv2, 'dnn'):
            print("DNN module imported successfully")
    except Exception as e:
        print(f"DNN module error: {e}")


def check_camera_access():
    """Check camera accessibility"""
    print("\n=== Camera Access Check ===")
    
    # Check for camera devices
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("Available camera devices:")
            print(result.stdout)
        else:
            print("Could not list camera devices")
    except FileNotFoundError:
        print("v4l2-ctl not found. Install with: sudo apt-get install v4l-utils")
    except Exception as e:
        print(f"Error checking camera devices: {e}")
    
    # Test basic camera access
    print("Testing camera access...")
    try:
        # Try different camera indices
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Camera {i}: SUCCESS - Frame shape {frame.shape}")
                else:
                    print(f"Camera {i}: OPENED but no frames")
                cap.release()
            else:
                print(f"Camera {i}: FAILED to open")
    except Exception as e:
        print(f"Camera test error: {e}")


def check_gstreamer():
    """Check GStreamer availability"""
    print("\n=== GStreamer Check ===")
    
    try:
        # Test GStreamer pipeline
        pipeline = "videotestsrc ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("GStreamer test: SUCCESS")
            else:
                print("GStreamer opened but no frames")
            cap.release()
        else:
            print("GStreamer test: FAILED")
    except Exception as e:
        print(f"GStreamer test error: {e}")


def test_opencv_operations():
    """Test basic OpenCV operations that might cause illegal instructions"""
    print("\n=== OpenCV Operations Test ===")
    
    try:
        # Test basic operations
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        print("Basic NumPy array creation: OK")
        
        # Test image operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Color conversion: OK")
        
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        print("Gaussian blur: OK")
        
        edges = cv2.Canny(gray, 50, 150)
        print("Canny edge detection: OK")
        
        # Test DNN if available
        if hasattr(cv2, 'dnn'):
            try:
                # Very simple DNN test
                net = cv2.dnn.readNetFromDarknet("", "")  # This should fail gracefully
            except:
                print("DNN module accessible (expected failure for empty params)")
        
        print("All basic operations passed!")
        
    except Exception as e:
        print(f"Operation test failed: {e}")
        import traceback
        traceback.print_exc()


def check_model_files():
    """Check if model files exist"""
    print("\n=== Model Files Check ===")
    
    model_path = 'ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
    config_path = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
    
    print(f"Looking for model file: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    print(f"Looking for config file: {config_path}")
    print(f"File exists: {os.path.exists(config_path)}")
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"Model file size: {size} bytes ({size/1024/1024:.1f} MB)")


def suggest_solutions():
    """Provide suggestions based on findings"""
    print("\n=== Troubleshooting Suggestions ===")
    print("If you're getting 'Illegal instruction' errors:")
    print("1. Try the alternative version: jetson_human_detection_alt.py")
    print("2. Reinstall OpenCV with proper ARM compilation:")
    print("   pip uninstall opencv-python opencv-contrib-python")
    print("   pip install opencv-python==4.5.3.56")
    print("3. Check if you're using the correct Python interpreter")
    print("4. Try running with different optimization flags")
    print("5. Consider using pre-compiled Jetson-specific OpenCV packages")


def main():
    """Run all diagnostics"""
    print("Jetson Nano Human Detection Diagnostics")
    print("=" * 50)
    
    check_system_info()
    check_camera_access()
    check_gstreamer()
    test_opencv_operations()
    check_model_files()
    suggest_solutions()
    
    print("\n" + "=" * 50)
    print("Diagnostics complete!")


if __name__ == "__main__":
    main()