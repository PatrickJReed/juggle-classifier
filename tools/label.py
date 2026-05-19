"""Two-pass tap-to-mark juggle labeling tool.

Pass 1: video plays, user presses SPACE for every Left_Foot juggle.
Pass 2: same video, SPACE for every Right_Foot juggle.

Output appends to labels/<video_stem>.csv with columns (frame, foot).
If the labels file already exists for the current foot, those rows are
preserved unchanged when the other-foot pass writes its own.

Controls during playback:
  SPACE - mark juggle at current frame
  p     - pause / resume
  u     - undo last mark in current pass
  j     - slow down (multiply speed by 0.75)
  k     - speed up (multiply speed by 1/0.75)
  h     - seek -2s
  l     - seek +2s
  q     - quit and save
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

SPACE, P, U, J, K, H, L, Q = (ord(c) for c in " pujkhlq")

HUD_HEIGHT = 110


def _compose_display(frame, *, foot, frame_idx, total, paused, speed, marks_count, last_mark):
    """Stack a HUD strip above the frame so the video itself is never occluded."""
    h, w = frame.shape[:2]
    hud = np.zeros((HUD_HEIGHT, w, 3), dtype=np.uint8)
    color_foot = (0, 200, 255) if foot == 'Left_Foot' else (255, 200, 0)
    lines = [
        f"Pass: {foot}   Marks: {marks_count}   Last: {last_mark if last_mark is not None else '-'}",
        f"Frame: {frame_idx}/{total - 1}   Speed: {speed:.2f}x   {'PAUSED' if paused else 'PLAYING'}",
        "SPACE=mark  p=pause  u=undo  j/k=slower/faster  h/l=seek-2s/+2s  q=quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(hud, line, (10, 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_foot if i == 0 else (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([hud, frame])


def _load_existing(out_csv: Path) -> list[tuple[int, str]]:
    if not out_csv.exists():
        return []
    rows: list[tuple[int, str]] = []
    with out_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append((int(row['frame']), row['foot']))
            except (KeyError, ValueError):
                continue
    return rows


def _save(out_csv: Path, rows: list[tuple[int, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'foot'])
        writer.writerows(rows_sorted)


def label_pass(video: Path, out_csv: Path, foot: str, initial_speed: float = 0.5) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    existing = _load_existing(out_csv)
    other_foot_rows = [r for r in existing if r[1] != foot]
    this_foot_rows: list[tuple[int, str]] = [r for r in existing if r[1] == foot]
    initial_count = len(this_foot_rows)

    speed = initial_speed
    paused = False
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()

    win = f"Label {foot}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    last_render = time.time()

    while ok:
        target_dt = 1.0 / max(fps * speed, 1e-3)
        now = time.time()
        if not paused and (now - last_render) >= target_dt:
            frame_idx += 1
            if frame_idx >= total:
                break
            ok, frame = cap.read()
            if not ok:
                break
            last_render = now

        display = _compose_display(
            frame, foot=foot, frame_idx=frame_idx, total=total,
            paused=paused, speed=speed,
            marks_count=len(this_foot_rows),
            last_mark=this_foot_rows[-1][0] if this_foot_rows else None,
        )
        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 0xFF:
            continue
        if key == Q:
            break
        if key == SPACE:
            this_foot_rows.append((frame_idx, foot))
        elif key == P:
            paused = not paused
        elif key == U and this_foot_rows:
            this_foot_rows.pop()
        elif key == J:
            speed = max(0.1, speed * 0.75)
        elif key == K:
            speed = min(4.0, speed / 0.75)
        elif key == H:
            frame_idx = max(0, frame_idx - int(2 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            last_render = time.time()  # reset timer so the new frame holds a full tick
        elif key == L:
            frame_idx = min(total - 1, frame_idx + int(2 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            last_render = time.time()

    cap.release()
    cv2.destroyAllWindows()

    combined = other_foot_rows + this_foot_rows
    _save(out_csv, combined)
    print(f"\n{foot} pass: {len(this_foot_rows) - initial_count:+d} marks "
          f"(total this foot: {len(this_foot_rows)})  ->  {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-pass juggle labeler.")
    parser.add_argument('video', type=str)
    parser.add_argument('--foot', choices=['Left_Foot', 'Right_Foot', 'both'], default='both',
                        help="Which pass to run. 'both' runs Left_Foot then Right_Foot. Default: both.")
    parser.add_argument('--out', type=str, default=None,
                        help="Output CSV path. Default: labels/<video_stem>.csv")
    parser.add_argument('--speed', type=float, default=0.5, help="Initial playback speed multiplier.")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    out_csv = Path(args.out) if args.out else (Path('labels') / f"{video.stem}.csv")

    if args.foot in ('Left_Foot', 'both'):
        print(f"\n=== Pass 1: Left_Foot ===  press SPACE every left-foot juggle, q to finish")
        label_pass(video, out_csv, foot='Left_Foot', initial_speed=args.speed)
    if args.foot in ('Right_Foot', 'both'):
        print(f"\n=== Pass 2: Right_Foot ===  press SPACE every right-foot juggle, q to finish")
        label_pass(video, out_csv, foot='Right_Foot', initial_speed=args.speed)


if __name__ == '__main__':
    main()
