# Forklift Overspeeding Detection

This project detects forklifts in video, estimates their real-world movement speed, and identifies overspeeding using:

- Ultralytics YOLO object detection and tracking
- Depth Anything V2 metric depth estimation
- Automatic focal-length calibration
- Homography-based image-to-world coordinate conversion
- Exponential moving average speed smoothing

## Requirements

- Windows
- Python 3.9+
- OpenCV
- NumPy
- PyTorch
- Transformers
- Ultralytics
- Pillow

Install the dependencies:

````powershell
pip install opencv-python numpy torch transformers ultralytics pillow