#!/usr/bin/env python3
"""
Human Detection using SSD-MobileNet-v2 on Jetson Nano with CSI Camera (IMX477)
This program captures video from the CSI camera and uses SSD-MobileNet-v2 to detect humans.
"""

import cv2
import numpy as np
import sys
import time
import os

# Check if we're running on Jetson Nano
def is_jetson_nano():
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            return 'jetson' in model.lower()
    except:
        return False


def create_camera_pipeline(width=1280, height=720, fps=30):
    """
    Create GStreamer pipeline for CSI camera on Jetson Nano
    """
    pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink"
    )
    return pipeline


class HumanDetector:
    def __init__(self, model_path='ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb', 
                 config_path='ssd_mobilenet_v2_coco_2018_03_29.pbtxt', 
                 confidence_threshold=0.5):
        """
        Initialize the human detector with SSD-MobileNet-v2 model
        """
        self.confidence_threshold = confidence_threshold
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"Error: Model files not found!")
            print(f"Looking for:")
            print(f"  Model: {model_path}")
            print(f"  Config: {config_path}")
            print("Please run download_model.py first to download the model files.")
            sys.exit(1)
        
        # Load the model with error handling
        try:
            print("Loading SSD-MobileNet-v2 model...")
            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            # Enable CUDA backend if available (Jetson Nano optimization)
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA backend for GPU acceleration")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU backend")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative approach...")
            try:
                # Fallback method
                self.net = cv2.dnn_DetectionModel(model_path, config_path)
                self.net.setInputSize(300, 300)
                self.net.setInputScale(1.0 / 127.5)
                self.net.setInputMean((127.5, 127.5, 127.5))
                self.net.setInputSwapRB(True)
                print("Model loaded with fallback method.")
            except Exception as e2:
                print(f"Fallback method also failed: {e2}")
                print("Make sure you have the correct model files and OpenCV with DNN support.")
                sys.exit(1)
        
        # COCO dataset class labels (index 0 is background, index 1 is person)
        self.class_labels = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                            'teddy bear', 'hair drier', 'toothbrush']
    
    def detect_humans(self, frame):
        """
        Detect humans in the given frame
        """
        try:
            # Prepare frame for detection
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # Run detection
            detections = self.net.forward()
            
            human_boxes = []
            human_confidences = []
            
            # Parse detections (SSD format)
            for detection in detections[0, 0]:
                confidence = float(detection[2])
                class_id = int(detection[1])
                
                if confidence > self.confidence_threshold and class_id == 1:  # Person class
                    # Extract bounding box coordinates
                    h, w = frame.shape[:2]
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)
                    
                    # Convert to width/height format
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    human_boxes.append([x1, y1, box_width, box_height])
                    human_confidences.append(confidence)
            
            return human_boxes, human_confidences
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [], []


def main():
    """
    Main function to initialize camera and run object detection
    """
    print("=== Jetson Nano Human Detection System ===")
    print(f"Running on Jetson Nano: {is_jetson_nano()}")
    
    # Check OpenCV version and DNN support
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV DNN module available: {cv2.dnn}")
    
    try:
        # Initialize human detector
        print("\nInitializing object detection model...")
        detector = HumanDetector()
        
        # Create camera pipeline
        print("\nSetting up CSI camera...")
        camera_pipeline = create_camera_pipeline()
        print(f"GStreamer pipeline: {camera_pipeline}")
        
        # Create VideoCapture object
        print("Opening camera...")
        cap = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            print("Troubleshooting steps:")
            print("1. Check if camera is properly connected to CSI port")
            print("2. Verify camera drivers are installed")
            print("3. Try running: v4l2-ctl --list-devices")
            print("4. Check power supply to camera")
            return
        
        print("Camera initialized successfully!")
        print("\nStarting real-time human detection...")
        print("Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break
            
            frame_count += 1
            
            # Detect humans in the frame
            human_boxes, human_confidences = detector.detect_humans(frame)
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Draw bounding boxes around detected humans
            for i in range(len(human_boxes)):
                box = human_boxes[i]
                confidence = human_confidences[i]
                
                # Extract coordinates
                x, y, w, h = box
                
                # Draw rectangle around the human
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label and confidence
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display FPS and detection count
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Humans detected: {len(human_boxes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame with bounding boxes
            cv2.imshow('Human Detection - CSI Camera', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Resources released and program terminated.")


if __name__ == "__main__":
    main()