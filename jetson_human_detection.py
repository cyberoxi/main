#!/usr/bin/env python3
"""
Human Detection using SSD-MobileNet-v2 on Jetson Nano with CSI Camera (IMX477)
This program captures video from the CSI camera and uses SSD-MobileNet-v2 to detect humans.
"""

import cv2
import numpy as np
import sys
import time


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
        
        # Load the model
        try:
            self.net = cv2.dnn_DetectionModel(model_path, config_path)
            self.net.setInputSize(300, 300)
            self.net.setInputScale(1.0 / 127.5)
            self.net.setInputMean((127.5, 127.5, 127.5))
            self.net.setInputSwapRB(True)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have the model files in the correct location.")
            print("You can download them from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API")
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
        # Run object detection
        class_ids, confidences, boxes = self.net.detect(frame, confThreshold=self.confidence_threshold)
        
        # Filter for humans only (class ID 1 is person in COCO dataset)
        human_boxes = []
        human_confidences = []
        
        if len(class_ids) > 0:
            for i in range(len(class_ids)):
                if class_ids[i][0] == 1:  # Class ID 1 corresponds to person
                    box = boxes[i]
                    confidence = confidences[i][0]
                    human_boxes.append(box)
                    human_confidences.append(confidence)
        
        return human_boxes, human_confidences


def main():
    """
    Main function to initialize camera and run object detection
    """
    print("Initializing CSI camera and object detection model...")
    
    # Initialize human detector
    detector = HumanDetector()
    
    # Create camera pipeline
    camera_pipeline = create_camera_pipeline()
    print(f"Using pipeline: {camera_pipeline}")
    
    # Create VideoCapture object
    cap = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Make sure your CSI camera is properly connected and drivers are installed.")
        return
    
    print("Camera initialized successfully. Starting video feed with human detection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break
            
            # Detect humans in the frame
            human_boxes, human_confidences = detector.detect_humans(frame)
            
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
            
            # Display the frame with bounding boxes
            cv2.imshow('Human Detection - CSI Camera', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")


if __name__ == "__main__":
    main()