"""Per-frame feature extractor for juggle classification.

Runs the existing JuggleNet detection pipeline (YOLO + MediaPipe + Kalman) on
a video and writes one CSV row per frame with ~28 hand-crafted features.

For each tracked body part / ball we record:
  - the Kalman-smoothed XY position (used as the position feature)
  - a *_visible flag: 1 if the underlying detector returned a real measurement
    on that frame, 0 if the value is Kalman-extrapolated from prior frames.

This split is important: low-visibility frames are signal, not noise — they're
how the classifier learns to attribute juggles when one signal source briefly
drops out (back-to-camera, ball-out-of-frame).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Make sibling 'utils' package importable when run as a script.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.vision_estimate import get_POI  # noqa: E402
from utils.update_predict import update_measurements, predict_KF  # noqa: E402

PARTS = ['Ball', 'Head', 'Left_Knee', 'Right_Knee', 'Left_Foot', 'Right_Foot']


def _initial_history():
    return {part: np.empty(shape=(0, 4)) for part in PARTS}


def _is_visible(poi_val) -> bool:
    """True iff the raw detector returned a real (non-None, non-NaN) x coordinate."""
    x = poi_val[0]
    if x is None:
        return False
    try:
        return not np.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _xy(arr: np.ndarray) -> tuple[float, float]:
    """Latest Kalman-smoothed (x, y) for a part; NaN if no history yet."""
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(arr[-1, 0]), float(arr[-1, 1]))


def extract(video_path: Path, ball_conf: float, track: bool) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    measurements = _initial_history()
    predictions = _initial_history()
    rows: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pois = get_POI(frame, ball_conf=ball_conf, track=track)
            # Raw measurements drive _visible flags — check BEFORE Kalman update.
            raw_visible = {
                'Ball':       _is_visible(pois['Ball']),
                'Left_Foot':  _is_visible(pois['Left_Foot']),
                'Right_Foot': _is_visible(pois['Right_Foot']),
            }
            measurements = update_measurements(measurements, pois)
            predictions = predict_KF(measurements, predictions)

            ball_x, ball_y = _xy(predictions['Ball'])
            ball_w = float(predictions['Ball'][-1, 2]) if len(predictions['Ball']) else np.nan
            ball_h = float(predictions['Ball'][-1, 3]) if len(predictions['Ball']) else np.nan
            lfoot_x, lfoot_y = _xy(predictions['Left_Foot'])
            rfoot_x, rfoot_y = _xy(predictions['Right_Foot'])
            lknee_x, lknee_y = _xy(predictions['Left_Knee'])
            rknee_x, rknee_y = _xy(predictions['Right_Knee'])
            head_x, head_y = _xy(predictions['Head'])

            rows.append({
                'frame': frame_idx,
                'ball_x': ball_x, 'ball_y': ball_y, 'ball_w': ball_w, 'ball_h': ball_h,
                'ball_visible': int(raw_visible['Ball']),
                'lfoot_x': lfoot_x, 'lfoot_y': lfoot_y,
                'lfoot_visible': int(raw_visible['Left_Foot']),
                'rfoot_x': rfoot_x, 'rfoot_y': rfoot_y,
                'rfoot_visible': int(raw_visible['Right_Foot']),
                'lknee_x': lknee_x, 'lknee_y': lknee_y,
                'rknee_x': rknee_x, 'rknee_y': rknee_y,
                'head_x': head_x, 'head_y': head_y,
            })
            frame_idx += 1
    finally:
        cap.release()

    df = pd.DataFrame(rows)

    # Derived features. .diff() introduces NaN at row 0 — LightGBM handles NaN.
    df['ball_dy'] = df['ball_y'].diff()
    df['ball_dy2'] = df['ball_dy'].diff()
    df['lfoot_dy'] = df['lfoot_y'].diff()
    df['rfoot_dy'] = df['rfoot_y'].diff()
    df['lknee_dy'] = df['lknee_y'].diff()
    df['rknee_dy'] = df['rknee_y'].diff()
    df['lfoot_rfoot_dy'] = df['lfoot_y'] - df['rfoot_y']
    df['lfoot_rfoot_dist'] = np.sqrt(
        (df['lfoot_x'] - df['rfoot_x'])**2 + (df['lfoot_y'] - df['rfoot_y'])**2
    )
    df['ball_lfoot_dist'] = np.sqrt(
        (df['ball_x'] - df['lfoot_x'])**2 + (df['ball_y'] - df['lfoot_y'])**2
    )
    df['ball_rfoot_dist'] = np.sqrt(
        (df['ball_x'] - df['rfoot_x'])**2 + (df['ball_y'] - df['rfoot_y'])**2
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-frame features from a video.")
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--ball-conf', type=float, default=0.10,
                        help='YOLO confidence threshold for ball (default 0.10)')
    parser.add_argument('--no-track', action='store_true',
                        help='Disable ByteTrack temporal association (default: enabled)')
    args = parser.parse_args()

    video = Path(args.video)
    out = Path(args.output)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    print(f"Extracting features from {video} ...")
    df = extract(video, ball_conf=args.ball_conf, track=not args.no_track)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} frames -> {out}")


if __name__ == '__main__':
    main()
