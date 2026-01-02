# Human Detection on Jetson Nano with CSI Camera

This project provides a real-time human detection system using SSD-MobileNet-v2 on a Jetson Nano with a CSI camera (IMX477).

## Files Included

- `jetson_human_detection.py` - Main program with SSD-MobileNet-v2
- `jetson_human_detection_alt.py` - Alternative version using Haar cascades (more stable)
- `system_diagnostics.py` - Diagnostic tool for troubleshooting
- `download_model.py` - Script to download required model files
- `requirements.txt` - Python dependencies

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

### Option 1: Main Program (Recommended if it works)
```bash
python jetson_human_detection.py
```

### Option 2: Alternative Program (More compatible)
```bash
python jetson_human_detection_alt.py
```

### Option 3: Run diagnostics first
```bash
python system_diagnostics.py
```

## Troubleshooting "Illegal Instruction" Errors

If you encounter "Illegal instruction (core dumped)" errors:

1. **Try the alternative version first**:
   ```bash
   python jetson_human_detection_alt.py
   ```
   This version uses Haar cascades instead of deep learning models.

2. **Run diagnostics**:
   ```bash
   python system_diagnostics.py
   ```
   This will help identify the specific issue.

3. **Reinstall OpenCV with ARM-compatible version**:
   ```bash
   pip uninstall opencv-python opencv-contrib-python
   pip install opencv-python==4.5.3.56
   ```

4. **Check camera connection**:
   ```bash
   v4l2-ctl --list-devices
   ```

5. **Verify system information**:
   ```bash
   uname -a
   lsb_release -a
   ```

## Features

Both programs will:
- Initialize the CSI camera
- Display the camera feed with bounding boxes around detected humans
- Show confidence scores and detection count
- Display real-time FPS
- Allow quitting with 'q' key

## Configuration

You can adjust these parameters in the code:

- Camera resolution: Modify the `width` and `height` parameters
- Confidence threshold: Change the `confidence_threshold` parameter
- Camera flip method: Modify `flip-method` in the GStreamer pipeline

## Note

- The main program uses COCO dataset pre-trained SSD-MobileNet-v2 model
- The alternative program uses Haar cascades for better compatibility
- Class ID 1 in COCO dataset corresponds to "person"
- Lower resolutions provide better performance on Jetson Nano