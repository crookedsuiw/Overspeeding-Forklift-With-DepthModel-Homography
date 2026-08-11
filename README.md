# Forklift Overspeeding Detection using yolo

This project detects forklifts in video, estimates their real-world movement speed, and identifies overspeeding from a still frame monocular camera and inference efficiently on a cpu.

It works in 3 Phases:

    Phase 1: Calibrating camera intrinsics:
We run the depth model and yolo model on forklift frames to calibrate the camera intrinsics.

    Phase 2: Calibrating Homography Plane:
We use the depth model, camera intrinsics and ROI selection of the ground/path of the forklift to calibrate Homography.

    Phase 3: Inference
The depth model is dropped and inference is solely run on yolo model using homography.



## Install Requirements

'''bash
pip install uv
uv pip install requirements.txt
