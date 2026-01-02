#!/usr/bin/env python3
"""
Alternative Human Detection - Simplified version for Jetson Nano
This version uses a more compatible approach to avoid "Illegal instruction" errors.
"""

import cv2
import numpy as np
import sys
import time
import os


def create_camera_pipeline(width=640, height=480, fps=30):
    """
    Create GStreamer pipeline for CSI camera on Jetson Nano - simplified version
    """
    pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink"
    )
    return pipeline


class SimpleHumanDetector:
    def __init__(self, confidence_threshold=0.5):
        """
        Simple human detector using Haar cascades (more compatible with Jetson Nano)
        """
        self.confidence_threshold = confidence_threshold
        self.human_cascade = None
        
        # Try to load Haar cascade for full body detection
        cascade_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_fullbody.xml',
            '/usr/share/opencv/haarcascades/haarcascade_fullbody.xml',
            'haarcascade_fullbody.xml'
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                try:
                    self.human_cascade = cv2.CascadeClassifier(path)
                    print(f"Loaded Haar cascade from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load cascade from {path}: {e}")
        
        if self.human_cascade is None:
            print("Warning: Could not load Haar cascade classifier.")
            print("Falling back to motion detection method.")
    
    def detect_humans_haar(self, frame):
        """
        Detect humans using Haar cascades
        """
        if self.human_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        humans = self.human_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to the format expected by main program
        boxes = []
        confidences = []
        
        for (x, y, w, h) in humans:
            boxes.append([x, y, w, h])
            confidences.append(0.8)  # Fixed confidence for Haar detection
        
        return boxes, confidences
    
    def detect_humans_motion(self, frame, prev_frame=None):
        """
        Simple motion-based human detection
        """
        if prev_frame is None:
            return [], []
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate magnitude of motion
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Threshold for significant motion
        motion_mask = mag > 2.0
        
        # Find contours of moving regions
        contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        confidences = []
        
        # Filter contours by size (assume human-sized movements)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                # Reasonable human size constraints
                if w > 30 and h > 50 and h > w * 1.2:  # Height should be greater than width
                    boxes.append([x, y, w, h])
                    confidences.append(min(0.9, area / 10000))  # Normalize confidence
        
        return boxes, confidences


def main():
    """
    Main function with error handling for Jetson Nano
    """
    print("=== Jetson Nano Alternative Human Detection ===")
    
    # Check system info
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Initialize detectors
    print("\nInitializing detection methods...")
    haar_detector = SimpleHumanDetector()
    
    # Create camera pipeline with lower resolution for better performance
    print("\nSetting up camera with reduced resolution...")
    camera_pipeline = create_camera_pipeline(width=640, height=480, fps=15)
    print(f"Pipeline: {camera_pipeline}")
    
    # Try to open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Error: Could not open camera with GStreamer.")
        print("Trying fallback method with V4L2...")
        
        # Fallback to V4L2
        cap = cv2.VideoCapture(0)  # Try first camera device
        if not cap.isOpened():
            print("Error: Could not open any camera.")
            print("Please check:")
            print("1. Camera connection to CSI port")
            print("2. Camera drivers installation")
            print("3. Run: v4l2-ctl --list-devices")
            return
    
    print("Camera opened successfully!")
    
    # Test frame capture
    print("Testing frame capture...")
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Cannot read frames from camera.")
        cap.release()
        return
    
    print(f"Frame shape: {test_frame.shape}")
    
    prev_frame = None
    frame_count = 0
    start_time = time.time()
    
    print("\nStarting detection loop...")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break
            
            frame_count += 1
            
            # Method 1: Haar cascade detection
            haar_boxes, haar_confidences = haar_detector.detect_humans_haar(frame)
            
            # Method 2: Motion detection (if we have previous frame)
            motion_boxes, motion_confidences = [], []
            if prev_frame is not None:
                motion_boxes, motion_confidences = haar_detector.detect_humans_motion(frame, prev_frame)
            
            # Combine detections (prefer Haar if available)
            if len(haar_boxes) > 0:
                final_boxes = haar_boxes
                final_confidences = haar_confidences
                detection_method = "Haar Cascade"
            else:
                final_boxes = motion_boxes
                final_confidences = motion_confidences
                detection_method = "Motion Detection"
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Draw detections
            for i, box in enumerate(final_boxes):
                x, y, w, h = box
                confidence = final_confidences[i] if i < len(final_confidences) else 0.5
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label
                label = f"Human ({confidence:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add info overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Method: {detection_method}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {len(final_boxes)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Alternative Human Detection', frame)
            
            # Store current frame for next iteration
            prev_frame = frame.copy()
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Program terminated.")


if __name__ == "__main__":
    main()