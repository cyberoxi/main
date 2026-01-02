# Human Detection on Jetson Nano with CSI Camera

This project provides a real-time human detection system using SSD-MobileNet-v2 on a Jetson Nano with a CSI camera (IMX477).

## Prerequisites

Before running the program, make sure you have:

1. **Jetson Nano** with JetPack SDK installed
2. **CSI Camera** (IMX477) properly connected to the Jetson Nano
3. **Python 3.x** installed
4. **OpenCV** with DNN module support
5. **Numpy**

## Installation

1. Clone or copy this repository to your Jetson Nano

2. Install required Python packages:
   ```bash
   pip install opencv-python numpy
   ```

3. Download the SSD-MobileNet-v2 model files:
   ```bash
   python download_model.py
   ```

## Running the Program

1. Make sure your CSI camera is properly connected and powered

2. Run the human detection program:
   ```bash
   python jetson_human_detection.py
   ```

3. The program will:
   - Initialize the CSI camera
   - Load the SSD-MobileNet-v2 model
   - Display the camera feed with bounding boxes around detected humans
   - Show confidence scores for each detection

4. Press 'q' to quit the program

## Troubleshooting

- If the camera doesn't initialize, make sure it's properly connected and the drivers are installed
- If the model fails to load, ensure the model files are downloaded in the correct directory
- If you see no detections, try adjusting the confidence threshold in the code
- If you encounter performance issues, consider reducing the camera resolution in the code

## Configuration

You can adjust these parameters in the code:

- Camera resolution: Modify the `width` and `height` parameters in `create_camera_pipeline()`
- Confidence threshold: Change the `confidence_threshold` parameter in `HumanDetector.__init__()`
- Camera flip method: Modify `flip-method` in the GStreamer pipeline if needed

## Note

This program uses the COCO dataset pre-trained model, where class ID 1 corresponds to "person", which is what we're detecting as "human".