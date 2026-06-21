import cv2
import numpy as np
import pandas as pd
import time
import os
import metrics

# Try importing mediapipe, fallback to mock-only if it's missing or fails
try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class HandProcessor:
    def __init__(self, static_mode=False, max_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mp_available = MEDIAPIPE_AVAILABLE
        if self.mp_available:
            try:
                base_options = BaseOptions(model_asset_path='hand_landmarker.task')
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=max_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.landmarker = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                print(f"Error initializing MediaPipe HandLandmarker Tasks API: {e}")
                self.landmarker = None
                self.mp_available = False
        else:
            self.landmarker = None

    def process_frame(self, frame):
        """
        Process a single OpenCV frame to extract hand landmarks using the new Tasks API.
        """
        if not self.mp_available or self.landmarker is None:
            return frame, {}

        h, w, _ = frame.shape
        # Convert BGR to RGB and convert to mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            results = self.landmarker.detect(mp_image)
        except Exception as e:
            print(f"Error detecting hand landmarks: {e}")
            return frame, {}
        
        hand_data = {}
        annotated_frame = frame.copy()
        
        # Define connection lines for skeleton drawing
        CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8), # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17) # Knuckles
        ]

        if results.hand_landmarks and results.handedness:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Extract label: "Left" or "Right"
                handedness = results.handedness[idx][0]
                label = handedness.category_name  # "Left" or "Right"
                
                # Check confidence
                if handedness.score < 0.3:
                    continue

                # Extract key landmarks in pixel coordinates (using frame dimensions)
                landmarks_dict = {}
                for l_idx, lm in enumerate(hand_landmarks):
                    px_x = lm.x * w
                    px_y = lm.y * h
                    px_z = lm.z * w  # Depth scale approximation
                    landmarks_dict[l_idx] = np.array([px_x, px_y, px_z])
                
                # We need Wrist (0), Thumb Tip (4), Index Tip (8), Middle MCP (9)
                wrist = landmarks_dict[0]
                thumb_tip = landmarks_dict[4]
                index_tip = landmarks_dict[8]
                middle_mcp = landmarks_dict[9]
                
                hand_data[label] = {
                    "wrist": wrist,
                    "thumb_tip": thumb_tip,
                    "index_tip": index_tip,
                    "middle_mcp": middle_mcp,
                    "all": landmarks_dict
                }
                
                # Draw skeleton lines (white/blue hue)
                for start_idx, end_idx in CONNECTIONS:
                    if start_idx in landmarks_dict and end_idx in landmarks_dict:
                        start_pt = (int(landmarks_dict[start_idx][0]), int(landmarks_dict[start_idx][1]))
                        end_pt = (int(landmarks_dict[end_idx][0]), int(landmarks_dict[end_idx][1]))
                        cv2.line(annotated_frame, start_pt, end_pt, (255, 180, 100), 2)
                        
                # Draw circular joints
                for l_idx, pt in landmarks_dict.items():
                    # Green for Right, Orange/Red for Left
                    color = (0, 255, 128) if label == "Right" else (0, 100, 255)
                    cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 4, color, -1)
                
                # Draw text indicator above hand
                wrist_px = (int(wrist[0]), int(wrist[1]))
                cv2.putText(
                    annotated_frame, 
                    f"{label} Hand (Conf: {handedness.score:.2f})", 
                    (wrist_px[0] - 50, wrist_px[1] + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0) if label == "Right" else (255, 128, 0), 
                    2
                )
                
        return annotated_frame, hand_data


def generate_mock_hand_frame(width, height, frame_idx, total_frames=300):
    """
    Generate a high-end synthetic frame simulating hand movement for testing and demo.
    Returns:
        frame: BGR frame with synthetic hand drawn
        hand_data: Dictionary containing hand coordinates
    """
    # Create dark slate aesthetic background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw dark medical-style grid lines
    grid_spacing = 50
    for y in range(0, height, grid_spacing):
        cv2.line(frame, (0, y), (width, y), (15, 25, 35), 1)
    for x in range(0, width, grid_spacing):
        cv2.line(frame, (x, 0), (x, height), (15, 25, 35), 1)
        
    # Draw interface details
    cv2.putText(frame, "SIMULATING DEXTERITY VIDEO ANALYZER", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FRAME: {frame_idx:03d}/{total_frames} | SYSTEM: ONLINE", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1, cv2.LINE_AA)

    t = frame_idx / 30.0  # 30 fps
    hand_data = {}
    
    # ------------------ RIGHT HAND (HEALTHY & ACTIVE) ------------------
    # Right hand is active throughout the recording, doing fast periodic pinches.
    r_freq = 0.8 * 2 * np.pi  # 0.8 Hz
    r_pinch_factor = 0.5 + 0.5 * np.sin(r_freq * t)  # periodic factor [0, 1]
    
    r_wrist = np.array([int(width * 0.7), int(height * 0.7), 0.0])
    
    # Compute fingertip movements
    # When pinching (r_pinch_factor close to 1), fingers close up
    r_thumb_x = int(width * 0.68 + (1 - r_pinch_factor) * 50) + np.random.normal(0, 1)
    r_thumb_y = int(height * 0.50 - (1 - r_pinch_factor) * 10) + np.random.normal(0, 1)
    
    r_index_x = int(width * 0.68 - (1 - r_pinch_factor) * 50) + np.random.normal(0, 1)
    r_index_y = int(height * 0.50 + (1 - r_pinch_factor) * 10) + np.random.normal(0, 1)
    
    r_middle_mcp = np.array([int(width * 0.72), int(height * 0.55), 0.0])
    
    hand_data["Right"] = {
        "wrist": r_wrist,
        "thumb_tip": np.array([r_thumb_x, r_thumb_y, 0.0]),
        "index_tip": np.array([r_index_x, r_index_y, 0.0]),
        "middle_mcp": r_middle_mcp
    }
    
    # Draw Right Hand (Greenish/Cyan)
    r_color = (255, 180, 0) # Slate blue
    cv2.circle(frame, (int(r_wrist[0]), int(r_wrist[1])), 8, (0, 255, 128), -1)
    cv2.circle(frame, (int(r_middle_mcp[0]), int(r_middle_mcp[1])), 6, (0, 200, 255), -1)
    cv2.circle(frame, (int(r_thumb_x), int(r_thumb_y)), 7, (0, 255, 255), -1)
    cv2.circle(frame, (int(r_index_x), int(r_index_y)), 7, (0, 255, 255), -1)
    
    # Draw bones
    cv2.line(frame, (int(r_wrist[0]), int(r_wrist[1])), (int(r_middle_mcp[0]), int(r_middle_mcp[1])), (255, 255, 255), 2)
    cv2.line(frame, (int(r_middle_mcp[0]), int(r_middle_mcp[1])), (int(r_index_x), int(r_index_y)), (255, 255, 255), 2)
    cv2.line(frame, (int(r_wrist[0]), int(r_wrist[1])), (int(r_thumb_x), int(r_thumb_y)), (255, 255, 255), 2)
    # Pinch distance indicator line
    cv2.line(frame, (int(r_thumb_x), int(r_thumb_y)), (int(r_index_x), int(r_index_y)), (0, 255, 0), 1)
    
    cv2.putText(frame, "Right Hand (Healthy)", (int(r_wrist[0]) - 80, int(r_wrist[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1, cv2.LINE_AA)
    
    # ------------------ LEFT HAND (AFFECTED & COOLDOWN) ------------------
    # Left hand is only active during the middle third of the video (frames 80 to 220).
    # It does only one slow, shaky pinch.
    l_active = 80 <= frame_idx <= 220
    if l_active:
        l_freq = 0.25 * 2 * np.pi  # 0.25 Hz (very slow)
        # One pinch attempt: peak of sine wave occurs around the middle of the active window
        # Shift time so sine peaks at frame 150 (t = 5s)
        phase = (t - 2.66) / 4.66 * np.pi  # Maps [80, 220] frames ([2.66s, 7.33s]) to [0, pi]
        l_pinch_factor = np.sin(phase) if 0 <= phase <= np.pi else 0.0
        
        l_wrist = np.array([int(width * 0.3), int(height * 0.72), 0.0])
        
        # Simulated tremor: high frequency jitter (8 Hz) to represent shakiness (Jerk)
        tremor_amplitude = 8.0
        tremor_x = tremor_amplitude * np.sin(8.0 * 2 * np.pi * t) + np.random.normal(0, 2)
        tremor_y = tremor_amplitude * np.cos(8.0 * 2 * np.pi * t) + np.random.normal(0, 2)
        
        # Fingertip coordinates
        # Notice that they don't get as close as the right hand (poor accuracy)
        l_thumb_x = int(width * 0.32 - (1 - l_pinch_factor * 0.6) * 50) + tremor_x
        l_thumb_y = int(height * 0.52 - (1 - l_pinch_factor * 0.6) * 12) + tremor_y
        
        l_index_x = int(width * 0.32 + (1 - l_pinch_factor * 0.6) * 50) + tremor_x
        l_index_y = int(height * 0.52 + (1 - l_pinch_factor * 0.6) * 12) + tremor_y
        
        l_middle_mcp = np.array([int(width * 0.28), int(height * 0.57), 0.0])
        
        hand_data["Left"] = {
            "wrist": l_wrist,
            "thumb_tip": np.array([l_thumb_x, l_thumb_y, 0.0]),
            "index_tip": np.array([l_index_x, l_index_y, 0.0]),
            "middle_mcp": l_middle_mcp
        }
        
        # Draw Left Hand (Orange/Red to show impairment)
        cv2.circle(frame, (int(l_wrist[0]), int(l_wrist[1])), 8, (0, 100, 255), -1)
        cv2.circle(frame, (int(l_middle_mcp[0]), int(l_middle_mcp[1])), 6, (0, 140, 255), -1)
        cv2.circle(frame, (int(l_thumb_x), int(l_thumb_y)), 7, (0, 0, 255), -1)
        cv2.circle(frame, (int(l_index_x), int(l_index_y)), 7, (0, 0, 255), -1)
        
        # Draw bones
        cv2.line(frame, (int(l_wrist[0]), int(l_wrist[1])), (int(l_middle_mcp[0]), int(l_middle_mcp[1])), (255, 255, 255), 2)
        cv2.line(frame, (int(l_middle_mcp[0]), int(l_middle_mcp[1])), (int(l_index_x), int(l_index_y)), (255, 255, 255), 2)
        cv2.line(frame, (int(l_wrist[0]), int(l_wrist[1])), (int(l_thumb_x), int(l_thumb_y)), (255, 255, 255), 2)
        # Pinch distance indicator line
        cv2.line(frame, (int(l_thumb_x), int(l_thumb_y)), (int(l_index_x), int(l_index_y)), (0, 0, 255), 1)
        
        cv2.putText(frame, "Left Hand (Tremor/Impaired)", (int(l_wrist[0]) - 80, int(l_wrist[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1, cv2.LINE_AA)
    else:
        # Left Hand is completely inactive
        cv2.putText(frame, "Left Hand: NOT DETECTED (NON-USE)", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1, cv2.LINE_AA)
        
    return frame, hand_data


def process_video_stream(video_path=None, use_mock=False):
    """
    Generator that processes video and yields frames and cumulative metrics.
    If video_path is None or file doesn't exist, it defaults to mock mode.
    
    Yields:
    -------
    dict:
        {
            "frame": np.ndarray (RGB frame for Streamlit),
            "progress": float (0.0 to 1.0),
            "left_history": dict,
            "right_history": dict,
            "status": str
        }
    """
    # Decide if we need mock
    is_webcam = isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit())
    if use_mock or not video_path or (not is_webcam and not os.path.exists(video_path)):
        is_mock = True
    else:
        is_mock = False
        
    if is_mock:
        total_frames = 300
        fps = 30.0
        width, height = 800, 500
        
        # Accumulate coordinates to calculate rolling metrics
        left_wrist_hist, left_thumb_hist, left_index_hist, left_scale_hist = [], [], [], []
        right_wrist_hist, right_thumb_hist, right_index_hist, right_scale_hist = [], [], [], []
        
        left_detected_count = 0
        right_detected_count = 0
        
        for idx in range(total_frames):
            frame, hand_data = generate_mock_hand_frame(width, height, idx, total_frames)
            
            # Convert BGR (OpenCV default) to RGB (Streamlit default)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Record Right Hand
            if "Right" in hand_data:
                right_detected_count += 1
                right_wrist_hist.append(hand_data["Right"]["wrist"])
                right_thumb_hist.append(hand_data["Right"]["thumb_tip"])
                right_index_hist.append(hand_data["Right"]["index_tip"])
                # Scale = distance from wrist to middle mcp
                scale = np.linalg.norm(hand_data["Right"]["wrist"] - hand_data["Right"]["middle_mcp"])
                right_scale_hist.append(scale)
                
            # Record Left Hand
            if "Left" in hand_data:
                left_detected_count += 1
                left_wrist_hist.append(hand_data["Left"]["wrist"])
                left_thumb_hist.append(hand_data["Left"]["thumb_tip"])
                left_index_hist.append(hand_data["Left"]["index_tip"])
                scale = np.linalg.norm(hand_data["Left"]["wrist"] - hand_data["Left"]["middle_mcp"])
                left_scale_hist.append(scale)

            # Compute current metrics
            r_metrics = {}
            if len(right_wrist_hist) > 0:
                # Get current kinematics of right hand wrist
                r_kin = metrics.calculate_kinematics(np.array(right_wrist_hist), fps)
                r_pinch = metrics.calculate_pinch_accuracy(
                    np.array(right_thumb_hist), 
                    np.array(right_index_hist), 
                    np.array(right_scale_hist)
                )
                r_metrics = {**r_kin, **r_pinch, "active_time": right_detected_count / fps}
                
            l_metrics = {}
            if len(left_wrist_hist) > 0:
                l_kin = metrics.calculate_kinematics(np.array(left_wrist_hist), fps)
                l_pinch = metrics.calculate_pinch_accuracy(
                    np.array(left_thumb_hist), 
                    np.array(left_index_hist), 
                    np.array(left_scale_hist)
                )
                l_metrics = {**l_kin, **l_pinch, "active_time": left_detected_count / fps}
            
            # Mock delay to match actual frame rate (~30fps)
            time.sleep(0.02)
            
            yield {
                "frame": rgb_frame,
                "progress": (idx + 1) / total_frames,
                "left_metrics": l_metrics,
                "right_metrics": r_metrics,
                "hand_data": hand_data,
                "status": "Running Simulation Mode"
            }
            
    else:
        # Real OpenCV / MediaPipe processing
        if isinstance(video_path, str) and video_path.isdigit():
            video_path = int(video_path)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {video_path}")
            
        # Configure fps and total frames
        if isinstance(video_path, int):
            fps = 30.0
            total_frames = 300 # Limit to 10 seconds capture for live webcam
        else:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 300 # Fallback
            
        processor = HandProcessor()
        
        left_wrist_hist, left_thumb_hist, left_index_hist, left_scale_hist = [], [], [], []
        right_wrist_hist, right_thumb_hist, right_index_hist, right_scale_hist = [], [], [], []
        
        left_detected_count = 0
        right_detected_count = 0
        frame_idx = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Mirror horizontally for live webcam to match user motor expectation
            if isinstance(video_path, int):
                frame = cv2.flip(frame, 1)
                
            frame_idx += 1
            annotated_frame, hand_data = processor.process_frame(frame)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Track right hand
            if "Right" in hand_data:
                right_detected_count += 1
                right_wrist_hist.append(hand_data["Right"]["wrist"])
                right_thumb_hist.append(hand_data["Right"]["thumb_tip"])
                right_index_hist.append(hand_data["Right"]["index_tip"])
                scale = np.linalg.norm(hand_data["Right"]["wrist"] - hand_data["Right"]["middle_mcp"])
                right_scale_hist.append(scale)
                
            # Track left hand
            if "Left" in hand_data:
                left_detected_count += 1
                left_wrist_hist.append(hand_data["Left"]["wrist"])
                left_thumb_hist.append(hand_data["Left"]["thumb_tip"])
                left_index_hist.append(hand_data["Left"]["index_tip"])
                scale = np.linalg.norm(hand_data["Left"]["wrist"] - hand_data["Left"]["middle_mcp"])
                left_scale_hist.append(scale)

            # Compute rolling metrics
            r_metrics = {}
            if len(right_wrist_hist) > 0:
                r_kin = metrics.calculate_kinematics(np.array(right_wrist_hist), fps)
                r_pinch = metrics.calculate_pinch_accuracy(
                    np.array(right_thumb_hist), 
                    np.array(right_index_hist), 
                    np.array(right_scale_hist)
                )
                r_metrics = {**r_kin, **r_pinch, "active_time": right_detected_count / fps}
                
            l_metrics = {}
            if len(left_wrist_hist) > 0:
                l_kin = metrics.calculate_kinematics(np.array(left_wrist_hist), fps)
                l_pinch = metrics.calculate_pinch_accuracy(
                    np.array(left_thumb_hist), 
                    np.array(left_index_hist), 
                    np.array(left_scale_hist)
                )
                l_metrics = {**l_kin, **l_pinch, "active_time": left_detected_count / fps}
                
            progress = min(1.0, frame_idx / total_frames)
            
            yield {
                "frame": rgb_frame,
                "progress": progress,
                "left_metrics": l_metrics,
                "right_metrics": r_metrics,
                "hand_data": hand_data,
                "status": f"Analyzing: {frame_idx}/{total_frames} frames"
            }
            
        cap.release()


def process_video(video_path):
    """
    Process a video file using OpenCV and MediaPipe Hands.
    Extracts WRIST, THUMB_TIP, and INDEX_FINGER_TIP (x, y) coordinates in pixels for every frame.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        ['frame', 'hand_label', 'wrist_x', 'wrist_y', 'thumb_x', 'thumb_y', 'index_x', 'index_y']
    """
    # Check if the file is mock or doesn't exist
    is_webcam = isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit())
    if not video_path or (not is_webcam and not os.path.exists(video_path)):
        data = []
        total_frames = 300
        width, height = 800, 500
        for f in range(total_frames):
            _, hand_data = generate_mock_hand_frame(width, height, f, total_frames)
            for hand, lm_dict in hand_data.items():
                data.append({
                    "frame": f,
                    "hand_label": hand,
                    "wrist_x": float(lm_dict["wrist"][0]),
                    "wrist_y": float(lm_dict["wrist"][1]),
                    "thumb_x": float(lm_dict["thumb_tip"][0]),
                    "thumb_y": float(lm_dict["thumb_tip"][1]),
                    "index_x": float(lm_dict["index_tip"][0]),
                    "index_y": float(lm_dict["index_tip"][1])
                })
        return pd.DataFrame(data)

    if isinstance(video_path, str) and video_path.isdigit():
        video_path = int(video_path)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
        
    processor = HandProcessor()
    frame_idx = 0
    records = []
    
    limit = 300 if isinstance(video_path, int) else None
    
    while True:
        if limit and frame_idx >= limit:
            break
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame, hand_data = processor.process_frame(frame)
        for hand, lm_dict in hand_data.items():
            records.append({
                "frame": frame_idx,
                "hand_label": hand,
                "wrist_x": float(lm_dict["wrist"][0]),
                "wrist_y": float(lm_dict["wrist"][1]),
                "thumb_x": float(lm_dict["thumb_tip"][0]),
                "thumb_y": float(lm_dict["thumb_tip"][1]),
                "index_x": float(lm_dict["index_tip"][0]),
                "index_y": float(lm_dict["index_tip"][1])
            })
        frame_idx += 1
        
    cap.release()
    return pd.DataFrame(records)
