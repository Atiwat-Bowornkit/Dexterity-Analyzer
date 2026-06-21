import numpy as np
from scipy.signal import savgol_filter

def calculate_speed(wrist_coords, fps):
    """
    Calculate the frame-to-frame velocity of the wrist.
    
    Parameters:
    -----------
    wrist_coords : array-like of shape (N, 2) or (N, 3)
        List or array of wrist coordinates.
    fps : float
        Frames per second of the video.
        
    Returns:
    --------
    float
        Average speed.
    np.ndarray
        Time-series array of speed at each frame.
    """
    coords = np.array(wrist_coords)
    N = len(coords)
    if N < 2:
        return 0.0, np.zeros(N)
    
    dt = 1.0 / fps
    # Using np.gradient to calculate stable central differences velocity vectors
    v_vectors = np.gradient(coords, dt, axis=0)
    speeds = np.linalg.norm(v_vectors, axis=1)
    
    avg_speed = float(np.mean(speeds))
    return avg_speed, speeds


def calculate_accuracy(thumb_coords, index_coords):
    """
    Calculate the Euclidean distance between the thumb tip and index finger tip (Pinch Grip distance).
    
    Parameters:
    -----------
    thumb_coords : array-like of shape (N, 2) or (N, 3)
        Thumb tip coordinates.
    index_coords : array-like of shape (N, 2) or (N, 3)
        Index finger tip coordinates.
        
    Returns:
    --------
    float
        Average pinch distance.
    np.ndarray
        Time-series array of pinch distance at each frame.
    """
    thumb = np.array(thumb_coords)
    index = np.array(index_coords)
    N = len(thumb)
    if N == 0:
        return 0.0, np.zeros(0)
        
    distances = np.linalg.norm(thumb - index, axis=1)
    avg_distance = float(np.mean(distances))
    return avg_distance, distances


def calculate_jerk(wrist_coords, fps):
    """
    Calculate the Jerk (rate of change of acceleration, or the third derivative of position) of the wrist.
    
    Parameters:
    -----------
    wrist_coords : array-like of shape (N, 2) or (N, 3)
        Wrist coordinates.
    fps : float
        Frames per second of the video.
        
    Returns:
    --------
    float
        Average jerk.
    np.ndarray
        Time-series array of jerk at each frame.
    """
    coords = np.array(wrist_coords)
    N = len(coords)
    if N < 4:
        return 0.0, np.zeros(N)
        
    dt = 1.0 / fps
    
    # Smooth coordinates using Savitzky-Golay filter to mitigate camera tracking jitter
    window_length = min(11, N)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 5:
        smoothed = np.zeros_like(coords)
        for i in range(coords.shape[1]):
            smoothed[:, i] = savgol_filter(coords[:, i], window_length, polyorder=3)
    else:
        smoothed = coords
        
    # Velocity, Acceleration, Jerk derivatives via gradient
    v = np.gradient(smoothed, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    
    jerk_magnitudes = np.linalg.norm(j, axis=1)
    avg_jerk = float(np.mean(jerk_magnitudes))
    return avg_jerk, jerk_magnitudes


def calculate_kinematics(positions, fps):
    """
    Calculate kinematic metrics (speed, acceleration, jerk) from landmark positions.
    """
    N = len(positions)
    if N < 4:
        return {
            "speed": np.zeros(N).tolist(),
            "acceleration": np.zeros(N).tolist(),
            "jerk": np.zeros(N).tolist(),
            "avg_speed": 0.0,
            "peak_speed": 0.0,
            "avg_acceleration": 0.0,
            "avg_jerk": 0.0,
            "smoothness_score": 100.0
        }
        
    avg_speed, speed = calculate_speed(positions, fps)
    avg_jerk, jerk = calculate_jerk(positions, fps)
    
    # Still calculate acceleration array
    dt = 1.0 / fps
    v_vectors = np.gradient(positions, dt, axis=0)
    a_vectors = np.gradient(v_vectors, dt, axis=0)
    acceleration = np.linalg.norm(a_vectors, axis=1)
    avg_acceleration = float(np.mean(acceleration))
    
    peak_speed = float(np.max(speed))
    
    # Log-dimensionless clinical scaling for smoothness score (0-100)
    jerk_val = max(1e-6, avg_jerk)
    smoothness_score = max(0.0, min(100.0, 100.0 - 20.0 * np.log10(jerk_val + 1.0)))
    
    return {
        "speed": speed.tolist(),
        "acceleration": acceleration.tolist(),
        "jerk": jerk.tolist(),
        "avg_speed": avg_speed,
        "peak_speed": peak_speed,
        "avg_acceleration": avg_acceleration,
        "avg_jerk": avg_jerk,
        "smoothness_score": float(smoothness_score)
    }


def calculate_pinch_accuracy(thumb_tips, index_tips, hand_scales):
    """
    Calculate the pinch distance and normalized accuracy over time.
    """
    N = len(thumb_tips)
    if N == 0:
        return {
            "pinch_distances": [],
            "norm_pinch_distances": [],
            "accuracy_over_time": [],
            "avg_accuracy": 0.0,
            "min_pinch_distance": 1.0
        }
        
    avg_dist, pinch_distances = calculate_accuracy(thumb_tips, index_tips)
    
    # Normalize by hand size to be invariant to camera distance
    safe_scales = np.where(hand_scales == 0, 1.0, hand_scales)
    norm_pinch_distances = pinch_distances / safe_scales
    
    accuracy_over_time = np.clip(1.0 - norm_pinch_distances, 0.0, 1.0)
    avg_accuracy = float(np.mean(accuracy_over_time))
    min_pinch_distance = float(np.min(norm_pinch_distances))
    
    return {
        "pinch_distances": pinch_distances.tolist(),
        "norm_pinch_distances": norm_pinch_distances.tolist(),
        "accuracy_over_time": accuracy_over_time.tolist(),
        "avg_accuracy": avg_accuracy,
        "min_pinch_distance": min_pinch_distance
    }


def detect_learned_non_use(left_metrics, right_metrics):
    """
    Analyze relative kinematics of Left vs Right hand to detect Learned Non-Use (LNU).
    LNU is characterized by a significant asymmetry where the affected hand is rarely used,
    or is significantly slower, less accurate, and jerkier than the unaffected hand,
    disproportionate to the physical neurological capacity.
    
    Parameters:
    -----------
    left_metrics : dict
        Kinematic and accuracy metrics of the Left hand.
    right_metrics : dict
        Kinematic and accuracy metrics of the Right hand.
        
    Returns:
    --------
    dict
        Result containing asymmetry indexes, detection decision, and detailed diagnostic explanation.
    """
    # Active ratio: frames where hand was detected / total frames
    # Let's assume this is passed or calculated.
    l_active = left_metrics.get("active_time", 0.0)
    r_active = right_metrics.get("active_time", 0.0)
    
    l_speed = left_metrics.get("avg_speed", 0.0)
    r_speed = right_metrics.get("avg_speed", 0.0)
    
    l_smooth = left_metrics.get("smoothness_score", 100.0)
    r_smooth = right_metrics.get("smoothness_score", 100.0)
    
    l_acc = left_metrics.get("avg_accuracy", 0.0)
    r_acc = right_metrics.get("avg_accuracy", 0.0)

    # Hand use ratio: Left Use vs Right Use
    total_active = l_active + r_active
    if total_active > 0:
        l_use_ratio = l_active / total_active
        r_use_ratio = r_active / total_active
    else:
        l_use_ratio = 0.5
        r_use_ratio = 0.5

    # Asymmetry index (range -1.0 to 1.0, where 0 is perfect symmetry)
    # AI = (R - L) / (R + L)
    def asymmetry_index(l_val, r_val):
        denom = (l_val + r_val)
        return (r_val - l_val) / denom if denom > 0 else 0.0

    speed_ai = asymmetry_index(l_speed, r_speed)
    use_ai = asymmetry_index(l_active, r_active)
    smooth_ai = asymmetry_index(l_smooth, r_smooth)
    acc_ai = asymmetry_index(l_acc, r_acc)

    # Determine Dominance Asymmetry & Learned Non-Use
    # If one hand has use_ai > 0.4 (i.e. one hand is used 70%+, other < 30%) and there is high kinematics discrepancy
    lnu_suspected = False
    affected_hand = None
    confidence = 0.0
    reason = "Motor movements are symmetric and within normal variability."

    if abs(use_ai) > 0.3:  # High asymmetry in hand use
        if use_ai > 0.3:
            # Right hand dominant use, Left hand may suffer from non-use
            affected_hand = "Left Hand"
            unaffected_hand = "Right Hand"
            discrepancy = (r_speed / l_speed) if l_speed > 0 else 10.0
            lnu_suspected = discrepancy > 1.5 or (r_smooth - l_smooth) > 15
            confidence = min(1.0, abs(use_ai) * 0.6 + (discrepancy / 5.0) * 0.4)
        else:
            # Left hand dominant use, Right hand may suffer from non-use
            affected_hand = "Right Hand"
            unaffected_hand = "Left Hand"
            discrepancy = (l_speed / r_speed) if r_speed > 0 else 10.0
            lnu_suspected = discrepancy > 1.5 or (l_smooth - r_smooth) > 15
            confidence = min(1.0, abs(use_ai) * 0.6 + (discrepancy / 5.0) * 0.4)
            
        if lnu_suspected:
            reason = (f"Learned Non-Use suspected in {affected_hand}. "
                      f"The patient demonstrates a severe bias towards the {unaffected_hand} (Use Ratio: {max(l_use_ratio, r_use_ratio):.1%}), "
                      f"and the {affected_hand} shows significantly lower speed (Asymmetry Index: {speed_ai:.2f}) "
                      f"and higher jerkiness (Smoothness difference: {abs(r_smooth - l_smooth):.1f} pts).")
        else:
            reason = (f"High asymmetry in hand use (bias towards {unaffected_hand}), but kinematic speed and "
                      f"smoothness are relatively comparable. Could be due to standard hand dominance or mild motor neglect.")
    elif abs(speed_ai) > 0.25:
        # Hands are used similarly in duration, but one is much slower
        slower_hand = "Left Hand" if speed_ai > 0 else "Right Hand"
        reason = f"Significant motor deceleration detected in the {slower_hand}, but active use time is still relatively symmetric."

    return {
        "lnu_suspected": lnu_suspected,
        "affected_hand": affected_hand,
        "confidence": confidence,
        "use_asymmetry_index": use_ai,
        "speed_asymmetry_index": speed_ai,
        "smoothness_asymmetry_index": smooth_ai,
        "accuracy_asymmetry_index": acc_ai,
        "left_use_ratio": l_use_ratio,
        "right_use_ratio": r_use_ratio,
        "diagnostic_summary": reason
    }
