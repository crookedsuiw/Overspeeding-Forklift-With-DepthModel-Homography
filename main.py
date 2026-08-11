import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO

# ==================================================================================
# HYPERPARAMETERS & CONFIGURATION
# ==================================================================================
YOLO_MODEL_WEIGHTS = "customForkliftn.pt" 
FORKLIFT_REAL_HEIGHT_M = 2
MIN_FORKLIFT_DETECTIONS = 5
YOLO_CONF_THRESH = 0.25
MIN_CALIBRATION_POINTS = 4
EMA_ALPHA = 0.15
MAX_FRAME_DIFF = 10
SCALE_FACTOR = 1.0 
SPEED_LIMIT = 1.5 # m/s
OVERSPEED_WINDOW = 20
TARGET_CLASS_ID = 0 # forklift/car classID

# ==================================================================================
# DEPTH MODEL WRAPPER
# ==================================================================================
class DepthModelWrapper:
    """
    Wrapper for Depth Anything V2 Metric using the transformers library.
    Ensure you have `transformers` and `torch` installed.
    """
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"): #depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
        import torch
        from transformers import pipeline

        print(f"[INFO] Loading Metric Depth Model: {model_id}...")
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the depth estimation pipeline
        self.depth_estimator = pipeline(
            task="depth-estimation",
            model=model_id,
            device=self.device
        )
        print("[SUCCESS] Depth model loaded.")

    def predict(self, frame):
        """
        Predicts the metric depth map for a given frame.
        Returns:
            depth_map (np.ndarray): 2D array of the same shape as frame (H, W)
                                    where each pixel value is the depth in meters (Z).
        """
        from PIL import Image

        # Convert OpenCV BGR frame to PIL Image (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Run inference
        result = self.depth_estimator(pil_image)

        # The pipeline returns a dictionary with 'depth' (a PIL Image or tensor depending on model)
        # and 'predicted_depth' (tensor). We extract the raw predicted depth map.
        depth_tensor = result["predicted_depth"]
        
        # Squeeze out batch/channel dims and convert to numpy
        depth_map = depth_tensor.squeeze().cpu().numpy()

        # Resize the depth map back to original frame dimensions (just in case the model downscaled it)
        H, W = frame.shape[:2]
        depth_map_resized = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

        return depth_map_resized

# ==================================================================================
# PIPELINE CLASS
# ==================================================================================
class AutoCalibrateSpeedPipeline:
    def __init__(self):
        print(f"[INFO] Loading YOLO model: {YOLO_MODEL_WEIGHTS}")
        self.yolo_model = YOLO(YOLO_MODEL_WEIGHTS)
        
        try:
            names_to_indices = {name: index for index, name in self.yolo_model.names.items()}
            self.target_class_idx = names_to_indices.get(TARGET_CLASS_ID, 0) # Default to 7 (truck in COCO)
        except Exception:
            self.target_class_idx = 0
            
        self.focal_length = None
        self.homography_matrix = None
        self.roi_polygon = None
        
        # Load from disk if exists
        if os.path.exists("focal_length.npy"):
            self.focal_length = float(np.load("focal_length.npy"))
            print(f"[INFO] Loaded focal length from disk: {self.focal_length:.2f}")
        if os.path.exists("homography_matrix.npy"):
            self.homography_matrix = np.load("homography_matrix.npy")
            print(f"[INFO] Loaded homography matrix from disk.")
        if os.path.exists("roi_polygon.npy"):
            self.roi_polygon = np.load("roi_polygon.npy")
            print(f"[INFO] Loaded ROI polygon from disk.")

    def phase_1_calibrate_focal_length(self, source_path):
        """
        Phase 1: Intrinsic Calibration (Focal Length Estimation)
        source_path: path to a folder of images or a video file.
        """
        print("\n--- PHASE 1: INTRINSIC CALIBRATION ---")
        
        if not hasattr(self, 'depth_model') or self.depth_model is None:
            self.depth_model = DepthModelWrapper()
            
        video_duration_seconds = 0
        if os.path.isfile(source_path) and not os.path.isdir(source_path):
            cap = cv2.VideoCapture(source_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                video_duration_seconds = frame_count / fps
            cap.release()

        frames = []
        focal_lengths = []
        
        while True:
            # CAPTURE STAGE
            if os.path.isdir(source_path):
                image_files = glob.glob(os.path.join(source_path, "*.[jp][pn]g"))
                for img_file in image_files:
                    frame = cv2.imread(img_file)
                    if frame is not None: frames.append(frame)
            elif os.path.isfile(source_path):
                cap = cv2.VideoCapture(source_path)
                window_name = "Phase 1: Scrub video and press 'c' to capture frame. 'q' to finish."
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
                print("--- INTERACTIVE FRAME SELECTION ---")
                print(f"Scrub through the video. Press 'c' when a forklift is clearly visible to capture the frame.")
                print(f"You need to capture enough frames to reach {MIN_FORKLIFT_DETECTIONS} accepted focal length samples.")
                print("Press 'q' when finished capturing this batch.")
                
                while True:
                    ret, frame = cap.read()
                    if not ret: 
                        print("Reached end of video.")
                        if video_duration_seconds < 15:
                            print("Video is shorter than 15s, restarting playback...")
                            cap.release()
                            cap = cv2.VideoCapture(source_path)
                            continue
                        else:
                            break
                    
                    disp = frame.copy()
                    cv2.putText(disp, f"Captured this round: {len(frames)} (Press 'c' to capture, 'q' to verify)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name, disp)
                    
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('c'):
                        frames.append(frame.copy())
                        print(f"Captured frame {len(frames)}.")
                    elif key == ord('q'):
                        break
                
                cv2.destroyWindow(window_name)
                cap.release()
            else:
                print("[ERROR] Invalid source path for Phase 1.")
                return False

            if not frames and not focal_lengths:
                print("[ERROR] No frames found or captured.")
                return False
                
            if not frames and focal_lengths:
                # User pressed 'q' without capturing any new frames this round
                pass 

            # VERIFICATION STAGE
            if frames:
                print("\n--- VERIFICATION STAGE ---")
                print("Press 'y' to accept a bounding box, 'n' to reject it.")
                
                unverified_frames = frames.copy()
                frames.clear() # clear for the next loop
                
                for frame in unverified_frames:
                    results = self.yolo_model.predict(frame, conf=YOLO_CONF_THRESH, classes=[self.target_class_idx], verbose=False)
                    if results and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            disp = frame.copy()
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(disp, f"Accept(y), Reject(n), Redraw(r)? Accepted: {len(focal_lengths)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            cv2.namedWindow("Verify Bounding Box", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("Verify Bounding Box", 1280, 720)
                            cv2.imshow("Verify Bounding Box", disp)
                            while True:
                                key = cv2.waitKey(0) & 0xFF
                                if key == ord('y'):
                                    pixel_height = y2 - y1
                                    cx = (x1 + x2) // 2
                                    cy = (y1 + y2) // 2
                                    depth_map = self.depth_model.predict(frame)
                                    depth_z = depth_map[cy, cx]
                                    f = (pixel_height * depth_z) / FORKLIFT_REAL_HEIGHT_M
                                    focal_lengths.append(f)
                                    print(f"Accepted box. Focal length: {f:.2f}")
                                    break
                                elif key == ord('n'):
                                    print("Rejected box.")
                                    break
                                elif key == ord('r'):
                                    print("Redrawing box... Draw a rectangle and press Space or Enter to confirm.")
                                    roi = cv2.selectROI("Verify Bounding Box", frame, fromCenter=False, showCrosshair=True)
                                    rx, ry, rw, rh = roi
                                    if rw > 0 and rh > 0:
                                        pixel_height = rh
                                        cx = rx + rw // 2
                                        cy = ry + rh // 2
                                        depth_map = self.depth_model.predict(frame)
                                        depth_z = depth_map[cy, cx]
                                        f = (pixel_height * depth_z) / FORKLIFT_REAL_HEIGHT_M
                                        focal_lengths.append(f)
                                        print(f"Accepted redrawn box. Focal length: {f:.2f}")
                                    else:
                                        print("Invalid ROI. Box rejected.")
                                    break
                try:
                    cv2.destroyWindow("Verify Bounding Box")
                except cv2.error:
                    pass

            # THRESHOLD CHECK
            if len(focal_lengths) >= MIN_FORKLIFT_DETECTIONS:
                break
            else:
                print(f"\n[WARNING] Only {len(focal_lengths)} valid detections accepted. Minimum required is {MIN_FORKLIFT_DETECTIONS}.")
                if os.path.isdir(source_path):
                    print("[ERROR] Failed Phase 1. Not enough detections in folder.")
                    return False
                print("Looping back to capture more frames...")

        self.focal_length = np.median(focal_lengths)
        np.save("focal_length.npy", self.focal_length)
        print(f"[SUCCESS] Focal length calibrated: {self.focal_length:.2f} pixels (based on {len(focal_lengths)} detections). Saved to focal_length.npy")
        return True

    def phase_2_generate_homography(self, video_path):
        """
        Phase 2: Interactive Auto-Generation of the Homography Matrix
        Allows user to select a static frame and pick points.
        """
        print("\n--- PHASE 2: HOMOGRAPHY GENERATION ---")
        if self.focal_length is None:
            print("[ERROR] Focal length not calibrated. Run Phase 1 first.")
            return False

        if not hasattr(self, 'depth_model') or self.depth_model is None:
            self.depth_model = DepthModelWrapper()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open {video_path}")
            return False

        # 1. Allow user to scrub and select a static frame
        static_frame = None
        window_name = "Select Static Frame (Press 's' to select, 'q' to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            disp = frame.copy()
            cv2.putText(disp, "Press 's' to select this frame as background", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, disp)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('s'):
                static_frame = frame.copy()
                break
            elif key == ord('q'):
                break
        
        cv2.destroyWindow(window_name)
        cap.release()

        if static_frame is None:
            print("[ERROR] No static frame selected.")
            return False

        # 2. Interactive Point Selection
        selected_pts_px = []
        depth_map = self.depth_model.predict(static_frame)
        
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # depth_norm = 255 - depth_norm  # Invert for non-metric visual consistency
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        
        def mouse_cb_main(event, x, y, flags, param):
            nonlocal inspected_pt
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_pts_px.append((x, y))
                print(f"Ground point selected: ({x}, {y})")
            elif event == cv2.EVENT_RBUTTONDOWN:
                inspected_pt = (x, y)
                z_val = depth_map[y, x]
                print(f"Inspected Depth at ({x}, {y}): {z_val:.2f}m")

        def mouse_cb_heatmap(event, x, y, flags, param):
            nonlocal inspected_pt
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                inspected_pt = (x, y)
                z_val = depth_map[y, x]
                print(f"Inspected Depth at ({x}, {y}): {z_val:.2f}m")

        window_name2 = "Select Floor Points (Press Enter when done)"
        cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name2, 1280, 720)
        cv2.namedWindow("Depth Map Heatmap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth Map Heatmap", 1280, 720)
        
        cv2.setMouseCallback(window_name2, mouse_cb_main)
        cv2.setMouseCallback("Depth Map Heatmap", mouse_cb_heatmap)

        inspected_pt = None

        while True:
            disp = static_frame.copy()
            disp_depth = depth_colormap.copy()
            cv2.putText(disp, f"L-Click: Select floor. R-Click: Inspect depth. ENTER: finish.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if len(selected_pts_px) > 1:
                pts = np.array(selected_pts_px, np.int32).reshape((-1, 1, 2))
                cv2.polylines(disp, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

            for pt in selected_pts_px:
                cv2.circle(disp, pt, 5, (0, 0, 255), -1)
                z_val = depth_map[pt[1], pt[0]]
                cv2.putText(disp, f"Z={z_val:.1f}m", (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
            if inspected_pt is not None:
                ix, iy = inspected_pt
                z_val = depth_map[iy, ix]
                cv2.circle(disp, (ix, iy), 5, (0, 255, 0), -1)
                cv2.putText(disp, f"INSPECT: Z={z_val:.1f}m", (ix+10, iy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(disp_depth, (ix, iy), 5, (0, 255, 0), -1)
                cv2.putText(disp_depth, f"INSPECT: Z={z_val:.1f}m", (ix+10, iy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            cv2.imshow(window_name2, disp)
            cv2.imshow("Depth Map Heatmap", disp_depth)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13: # Enter
                if len(selected_pts_px) >= MIN_CALIBRATION_POINTS:
                    break
                else:
                    print(f"Need at least {MIN_CALIBRATION_POINTS} points.")

        cv2.destroyWindow(window_name2)
        if cv2.getWindowProperty("Depth Map Heatmap", cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow("Depth Map Heatmap")

        # 3. 3D Projection & Homography
        H, W = static_frame.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        
        src_pts = np.float32(selected_pts_px).reshape(-1, 1, 2)
        dst_pts = []
        for (u, v) in selected_pts_px:
            Z = depth_map[v, u]
            X = ((u - cx) * Z) / self.focal_length
            Y = Z # Depth is Y on the floor plane
            dst_pts.append((X, Y))
            
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
        
        self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if self.homography_matrix is not None:
            np.save("homography_matrix.npy", self.homography_matrix)
            self.roi_polygon = np.array(selected_pts_px, np.int32)
            np.save("roi_polygon.npy", self.roi_polygon)
            print("[SUCCESS] Homography matrix and ROI generated successfully! Saved to disk.")
            return True
        else:
            print("[ERROR] Failed to generate homography matrix.")
            return False

    def phase_3_run_speed_estimation(self, video_path):
        """
        Phase 3: Real-Time Speed Estimation
        """
        print("\n--- PHASE 3: SPEED ESTIMATION ---")
        if self.homography_matrix is None:
            print("[ERROR] Homography matrix not found. Run Phase 2 first.")
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter('DetectionForkliftOverspeeding.mp4', fourcc, fps, (width, height))

        cv2.namedWindow("Auto-Calibrated Speed Estimation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Auto-Calibrated Speed Estimation", 1280, 720)

        object_positions = {}
        object_speeds = {}
        overspeed_history = {}
        overspeed_alerted = set()
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_num += 1

            disp = frame.copy()
            
            if self.roi_polygon is not None and len(self.roi_polygon) > 2:
                # Mark the points used for homography (draw polygon + circles at vertices)
                cv2.polylines(disp, [self.roi_polygon], True, (255, 0, 255), 2)
                for pt in self.roi_polygon:
                    cv2.circle(disp, tuple(pt), 4, (0, 255, 255), -1)
                
            results = self.yolo_model.track(frame, persist=True, verbose=False, classes=[self.target_class_idx, 1], tracker="botsort.yaml")

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                persons = []
                forklifts = []

                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    if cls_id == 1: # PERSON_CLASS_ID
                        px1, py1, px2, py2 = box
                        p_cx = (px1 + px2) / 2
                        p_cy = (py1 + py2) / 2
                        persons.append((p_cx, p_cy))
                    elif cls_id == self.target_class_idx:
                        forklifts.append((box, track_id))

                for box, track_id in forklifts:
                    x1, y1, x2, y2 = box
                    ground_x = (x1 + x2) / 2
                    ground_y = y2

                    for (p_cx, p_cy) in persons:
                        if x1 <= p_cx <= x2 and y1 <= p_cy <= y2:
                            ground_x = p_cx
                            break

                    # ROI filtering is disabled for the indoor script.
                    # if self.roi_polygon is not None and len(self.roi_polygon) > 2:
                    #     if cv2.pointPolygonTest(self.roi_polygon, (int(ground_x), int(ground_y)), False) < 0:
                    #         continue

                    feet_px = np.array([[ground_x, ground_y]], dtype=np.float32).reshape(-1, 1, 2)
                    
                    # Apply homography
                    world_coords = cv2.perspectiveTransform(feet_px, self.homography_matrix)
                    X, Y = world_coords[0][0]

                    if track_id in object_positions:
                        prev_X, prev_Y, prev_frame = object_positions[track_id]
                        frame_diff = frame_num - prev_frame

                        if 0 < frame_diff <= MAX_FRAME_DIFF:
                            distance_m = np.sqrt((X - prev_X)**2 + (Y - prev_Y)**2)
                            time_s = frame_diff / fps
                            speed_mps = distance_m / time_s

                            previous_speed = object_speeds.get(track_id, speed_mps)
                            smoothed_speed = (EMA_ALPHA * speed_mps) + (1 - EMA_ALPHA) * previous_speed
                            object_speeds[track_id] = smoothed_speed

                    object_positions[track_id] = (X, Y, frame_num)

                    # Default to not overspeeding if speed is not yet calculated
                    speed_mps = object_speeds.get(track_id, 0)
                    is_overspeeding = speed_mps > SPEED_LIMIT

                    if is_overspeeding:
                        if track_id not in overspeed_history:
                            overspeed_history[track_id] = []
                        overspeed_history[track_id].append(speed_mps)
                        
                        if len(overspeed_history[track_id]) >= OVERSPEED_WINDOW and track_id not in overspeed_alerted:
                            avg_speed = sum(overspeed_history[track_id]) / len(overspeed_history[track_id])
                            print(f"Overpeeding detected: ID: {track_id} ,speed: {avg_speed:.2f}")
                            overspeed_alerted.add(track_id)
                    else:
                        if track_id in overspeed_history:
                            overspeed_history[track_id].clear()
                        if track_id in overspeed_alerted:
                            overspeed_alerted.remove(track_id)

                    box_color = (0, 0, 255) if is_overspeeding else (0, 255, 0)
                    status_text = "Overspeeding" if is_overspeeding else "NOT"

                    cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    
                    if track_id in object_speeds:
                        label = f"ID:{track_id} {speed_mps:.1f} m/s - {status_text}"
                        cv2.putText(disp, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            cv2.imshow("Auto-Calibrated Speed Estimation", disp)
            out_writer.write(disp)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Output video saved as DetectionForkliftOverspeeding.mp4")


if __name__ == "__main__":
    # Example Usage:
    # 1. Initialize Pipeline
    pipeline = AutoCalibrateSpeedPipeline()
    
    # Define paths
    video_source = r"C:\Users\asimm\Downloads\JEZT\ForkliftOverspeeding\SyntheticForklift3.mp4"
    # Run the Pipeline end-to-end using the video source for all phases
    print("\n[STARTING PIPELINE]")
    
    # 2. Phase 1
    if pipeline.focal_length is not None or pipeline.phase_1_calibrate_focal_length(video_source):
        # 3. Phase 2
        if pipeline.homography_matrix is not None or pipeline.phase_2_generate_homography(video_source):
            # 4. Phase 3
            pipeline.phase_3_run_speed_estimation(video_source)
    
    print("\n[PIPELINE FINISHED]")
