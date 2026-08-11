# Forklift Overspeeding Detection using yolo

This project detects forklifts in video, estimates their real-world movement speed, and identifies overspeeding from a still frame monocular camera and inference efficiently on a cpu. Although we do use depth model, we use it only for calibration. Homography requires knowing the real world coodinates and distances. Instead we use the depth model to solve for the real world coordinates. We then estimate speed how homography-speed-estimation standardly does.

It works in 3 Phases:

    Phase 1: Calibrating camera intrinsics:
To estimate the camera intrinsics requires knowing the 2D image points and the real world coordinates.
Since we know the real world height of the forklift, we hardcode the value (in meters) FORKLIFT_REAL_HEIGHT_M.
We inference multiple frames for forklift bbox height, we run the depth model to get the depth in meters using DepthAnything-V2-metric. We have all variables to estimate camera intrinsics (except principle focus which is not required for speed estimation). Camera intrinsics are saved to focal_length.npy

    Phase 2: Calibrating Homography Plane:
We use the depth model, camera intrinsics and ROI selection of the path of the forklift to calibrate Homography. Saved to homography_matrix.npy

    Phase 3: Inference
The depth model is dropped and inference is solely run on yolo model using homography. Skips directly to phase 3 if focal_length.npy and homography_matrix.npy files are already available in the current path.

Download the Depth-Anything-V2-Metric-Indoor here:
https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf/


## Install Requirements

'''bash
pip install uv
uv pip install requirements.txt
