#!/usr/bin/env python3
"""
Script to download the SSD-MobileNet-v2 model files required for human detection
"""

import os
import urllib.request
import zipfile


def download_model():
    """
    Download the SSD-MobileNet-v2 model files from TensorFlow's model zoo
    """
    print("Downloading SSD-MobileNet-v2 model files...")
    
    # Create directory for model files
    model_dir = "ssd_mobilenet_v2_coco_2018_03_29"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Model URL
    model_url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
    
    # Download the model
    print("Downloading model archive...")
    try:
        urllib.request.urlretrieve(model_url, "ssd_mobilenet_v2_coco_2018_03_29.tar.gz")
        print("Model archive downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False
    
    # Extract the model
    print("Extracting model files...")
    try:
        import tarfile
        with tarfile.open("ssd_mobilenet_v2_coco_2018_03_29.tar.gz", "r:gz") as tar:
            tar.extractall()
        print("Model files extracted successfully.")
    except Exception as e:
        print(f"Error extracting model: {e}")
        return False
    
    # Remove the archive file
    os.remove("ssd_mobilenet_v2_coco_2018_03_29.tar.gz")
    
    # Rename the extracted folder if needed
    extracted_folder = "ssd_mobilenet_v2_coco_2018_03_29"
    if os.path.exists("ssd_mobilenet_v2_coco_2018_03_29") and not os.path.exists(model_dir):
        os.rename(extracted_folder, model_dir)
    
    print(f"Model files are now available in the '{model_dir}' directory.")
    return True


if __name__ == "__main__":
    success = download_model()
    if success:
        print("\nModel download completed successfully!")
        print("You can now run the human detection program.")
    else:
        print("\nModel download failed. Please check your internet connection and try again.")