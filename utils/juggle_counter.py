from scipy.signal import find_peaks
import numpy as np


class JuggleCounter:
    """
    Detects juggle events from ball trajectory using peak detection.

    Y-coordinate increases as you move lower down the screen. Therefore, 
    a minimum height in a ball trajectory, corresponding to a juggle kick,
    manifests as a maximum point in the Y-coordinate.

    We maintain state across frames to prevent double-counting and handle
    sliding window history.
    """

    def __init__(self,
                 prominence: float = 0.02,
                 min_gap_frames: int = 5,
                 max_distance: float = 0.3,
                 min_history_length: int = 10,
                 candidate_parts=None,
                 mode: str = 'ball',
                 foot_prominence: float = 0.02,
                 foot_validation_window: int = 5,
                 spread_prominence: float = 0.10):
        """
        Args:
            prominence: Minimum prominence for peak detection on ball Y
            min_gap_frames: Minimum frames between valid juggles
            max_distance: Maximum normalised distance for valid juggle attribution
            min_history_length: Minimum trajectory points needed for detection
            candidate_parts: Optional iterable of body-part names to restrict
                attribution to (e.g. ["Left_Foot", "Right_Foot"]). When None,
                any tracked body part may be credited.
            mode: 'ball' (default) detects peaks in ball Y; 'foot' detects
                peaks in foot Y and uses ball as a validator. Foot mode is
                robust to ball-out-of-frame moments.
            foot_prominence: Minimum prominence for peak detection on foot Y
                (only used in 'foot' mode).
        """

        self.prominence = prominence
        self.min_gap_frames = min_gap_frames
        self.max_distance = max_distance
        self.min_history_length = min_history_length
        self.candidate_parts = set(candidate_parts) if candidate_parts else None
        self.mode = mode
        self.foot_prominence = foot_prominence
        self.foot_validation_window = foot_validation_window
        self.spread_prominence = spread_prominence

        # State tracking
        self.last_counted_frame_number = -np.inf
        self.counts = {}
        self.total_frames_processed = 0

    def update(self, predictions: dict) -> dict:
        """
        Update juggle count based on current predictions

        Args:
            predictions: Dict of {POI: np.array(N, 4)}, with trajectory history
                         Format: [x, y, width, height] per row

        Returns:
            Updated self.counts dictionary
        """
        if self.mode == 'foot':
            return self._update_foot(predictions)
        if self.mode == 'spread':
            return self._update_foot_spread(predictions)
        return self._update_ball(predictions)

    def _update_ball(self, predictions: dict) -> dict:
        if "Ball" not in predictions or len(predictions["Ball"]) < self.min_history_length:
            self.total_frames_processed += 1
            return self.counts

        # Extract ball y-coordinates
        y = predictions["Ball"][:, 1]

        # Find all peaks in trajectory (Y inverted: peak = ball at lowest point = kick instant)
        peaks, properties = find_peaks(y, prominence=self.prominence)

        current_frame_number = self.total_frames_processed
        history_length = len(y)

        for peak_idx in peaks:
            peak_frame_number = current_frame_number - (history_length - 1 - peak_idx)
            if self.is_valid_peak(peak_frame_number):
                self.count_juggle_at_peak(peak_idx, predictions)
                self.last_counted_frame_number = peak_frame_number

        self.total_frames_processed += 1
        return self.counts

    def _update_foot(self, predictions: dict) -> dict:
        # Need at least one foot with sufficient history
        feet = [f for f in ('Left_Foot', 'Right_Foot')
                if f in predictions and len(predictions[f]) >= self.min_history_length]
        if not feet:
            self.total_frames_processed += 1
            return self.counts

        current_frame_number = self.total_frames_processed

        # Collect candidate (frame_number, foot, peak_idx) tuples across both feet
        candidates = []
        for foot in feet:
            y_foot = predictions[foot][:, 1]
            # Foot going UP -> Y decreases -> peak in -Y
            peaks, _ = find_peaks(-y_foot, prominence=self.foot_prominence)
            history_length = len(y_foot)
            for peak_idx in peaks:
                peak_frame_number = current_frame_number - (history_length - 1 - peak_idx)
                candidates.append((peak_frame_number, foot, peak_idx))

        # Process chronologically so dedupe across feet works
        candidates.sort(key=lambda c: c[0])
        for peak_frame_number, foot, peak_idx in candidates:
            if not self.is_valid_peak(peak_frame_number):
                continue
            if not self._validate_foot_peak_with_ball(predictions, peak_idx, foot):
                continue
            self.counts[foot] = self.counts.get(foot, 0) + 1
            self.last_counted_frame_number = peak_frame_number

        self.total_frames_processed += 1
        return self.counts

    def _update_foot_spread(self, predictions: dict) -> dict:
        # Detect kicks via the vertical spread between feet:
        # |L_Y - R_Y| is near zero when both feet are planted and peaks when one foot lifts.
        # At each peak the kicking foot is whichever has the smaller Y (higher on screen).
        if 'Left_Foot' not in predictions or 'Right_Foot' not in predictions:
            self.total_frames_processed += 1
            return self.counts

        L = predictions['Left_Foot'][:, 1]
        R = predictions['Right_Foot'][:, 1]
        n = min(len(L), len(R))
        if n < self.min_history_length:
            self.total_frames_processed += 1
            return self.counts

        L = L[-n:]
        R = R[-n:]
        spread = np.abs(L - R)
        # nan-safe: peaks in nan regions just won't be found
        peaks, _ = find_peaks(np.nan_to_num(spread, nan=0.0), prominence=self.spread_prominence)

        current_frame_number = self.total_frames_processed

        candidates = []
        for peak_idx in peaks:
            l_y = L[peak_idx]
            r_y = R[peak_idx]
            if np.isnan(l_y) or np.isnan(r_y):
                continue
            kicking_foot = 'Left_Foot' if l_y < r_y else 'Right_Foot'
            peak_frame_number = current_frame_number - (n - 1 - peak_idx)
            candidates.append((peak_frame_number, kicking_foot, peak_idx))

        candidates.sort(key=lambda c: c[0])
        for peak_frame_number, foot, peak_idx in candidates:
            if not self.is_valid_peak(peak_frame_number):
                continue
            if not self._validate_foot_peak_with_ball(predictions, peak_idx, foot):
                continue
            self.counts[foot] = self.counts.get(foot, 0) + 1
            self.last_counted_frame_number = peak_frame_number

        self.total_frames_processed += 1
        return self.counts

    def _validate_foot_peak_with_ball(self, predictions: dict, peak_idx: int, foot: str) -> bool:
        """
        Accept a foot peak if, within ±foot_validation_window frames of the peak,
        ANY frame shows either:
          - ball missing (NaN) — ball-out-of-frame recovery
          - ball within max_distance of the foot — ball-foot contact
        Reject only if every frame in the window has the ball visible but elsewhere
        (likely a step/walk while the ball is in the air or on the ground far from foot).

        The window accommodates the natural 1-3 frame lag between ball-at-lowest
        and foot-at-highest during a kick (foot follows through past contact).
        """
        if "Ball" not in predictions:
            return True
        ball_arr = predictions["Ball"]
        foot_arr = predictions[foot]
        n = len(ball_arr)
        if n == 0 or peak_idx >= n:
            return True

        lo = max(0, peak_idx - self.foot_validation_window)
        hi = min(n - 1, peak_idx + self.foot_validation_window)

        for i in range(lo, hi + 1):
            if i >= len(foot_arr):
                continue
            ball_pos = ball_arr[i, :2]
            if np.isnan(ball_pos[0]) or np.isnan(ball_pos[1]):
                return True
            foot_pos = foot_arr[i, :2]
            if np.isnan(foot_pos[0]) or np.isnan(foot_pos[1]):
                continue
            if np.linalg.norm(ball_pos - foot_pos) < self.max_distance:
                return True
        return False


    def is_valid_peak(self, peak_frame_number: int) -> bool:
        """Check if peak should be counted (not double-counted, sufficient time gap)"""
        return peak_frame_number > self.last_counted_frame_number + self.min_gap_frames
    
    def count_juggle_at_peak(self, peak_idx: int, predictions: dict):
        """Find closest body part at peak and increment count"""
        
        ball_pos = predictions["Ball"][peak_idx, :2]    # x, y coordinates

        min_dist = np.inf
        closest_part = None

        # Find closest body part
        for key, value in predictions.items():
            if key == "Ball" or len(value) <= peak_idx:
                continue
            if self.candidate_parts is not None and key not in self.candidate_parts:
                continue

            part_pos = value[peak_idx, :2]
            dist = np.linalg.norm(ball_pos - part_pos)

            if dist < min_dist:
                min_dist = dist
                closest_part = key
        
        # Only count if within distance threshold
        if closest_part and min_dist < self.max_distance:
            if closest_part not in self.counts:
                self.counts[closest_part] = 0
            self.counts[closest_part] += 1
    
    def reset(self):
        """Reset all counts and state"""
        self.counts = {}
        self.last_counted_peak_idx = np.inf
        self.frame_count = 0

    def get_counts(self) -> dict:
        """Get current juggle count"""
        return self.counts.copy()
    