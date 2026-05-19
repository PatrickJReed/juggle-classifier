import cv2
import argparse
import os
import numpy as np
import csv as _csv
from collections import defaultdict

from utils.vision_estimate import get_POI
from utils.juggle_counter import JuggleCounter
from utils.draw_POI import Visualiser
from utils.plot_graph import init_plot, update_plot
from utils.update_predict import update_measurements, predict_KF, predict_para
import matplotlib.pyplot as plt

POI = ["Ball", "Head", "Left_Knee", "Right_Knee", "Right_Foot", "Left_Foot"]


class EventListCounter:
    """
    Counter driven by a pre-computed events CSV. On each update() call we advance
    through the events sorted by frame and increment the per-foot count whenever
    the playback frame index has reached an event's frame.
    """

    def __init__(self, events_csv: str):
        rows: list[tuple[int, str]] = []
        with open(events_csv) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    rows.append((int(row['frame']), row['foot']))
                except (KeyError, ValueError):
                    continue
        self._events = sorted(rows, key=lambda r: r[0])
        self._next = 0
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, frame_idx: int) -> dict:
        while self._next < len(self._events) and self._events[self._next][0] <= frame_idx:
            _, foot = self._events[self._next]
            self._counts[foot] += 1
            self._next += 1
        return dict(self._counts)


def parse_args():
    parser = argparse.ArgumentParser(description="Football Juggle Counter")
    parser.add_argument('--video', type=str, default=None, help='Path to video file. Leave empty to use webcam.')
    parser.add_argument('--save', type=str, default=None, help='Path to save directory.')
    parser.add_argument('--plot', action='store_true', help='Plot ball Y-trajectory.')
    parser.add_argument('--save-plot', type=str, default=None, help='Path to save full-video ball Y-position plot (PNG).')
    parser.add_argument('--headless', action='store_true', help='Skip the OpenCV window (for running without a display).')
    parser.add_argument('--feet-only', action='store_true', help='Only attribute juggles to Left_Foot / Right_Foot (ignore knees and head).')
    parser.add_argument('--ball-conf', type=float, default=0.3, help='YOLO confidence threshold for ball detection (lower = more partial-ball detections, more false positives).')
    parser.add_argument('--track', action='store_true', help='Use YOLO + ByteTrack for temporal association across frames (better through partial occlusion / brief misses).')
    parser.add_argument('--foot-peaks', action='store_true', help='Count juggles via foot-Y peak detection (validated by ball when visible). Recovers juggles when ball is out of frame.')
    parser.add_argument('--foot-prominence', type=float, default=0.02, help='Prominence for foot-Y peak detection (only with --foot-peaks). Higher = stricter.')
    parser.add_argument('--foot-window', type=int, default=5, help='Validation window (frames) around each foot peak for ball-contact / ball-blackout check.')
    parser.add_argument('--foot-spread', action='store_true', help='Count juggles via |L_foot_Y - R_foot_Y| peak detection (asymmetric leg motion). Robust when back is to camera.')
    parser.add_argument('--spread-prominence', type=float, default=0.10, help='Prominence for inter-foot-spread peak detection (only with --foot-spread).')
    parser.add_argument('--events', type=str, default=None,
                        help='Path to a pre-computed events CSV (frame,foot,confidence). '
                             'When set, overrides the heuristic counter — the rendered count '
                             'reflects these events as the video plays through their frames.')

    return parser.parse_args()

def main():

    # parse arguments
    args = parse_args()

    # use non-interactive matplotlib backend when no live plot is needed
    if not args.plot:
        plt.switch_backend('Agg')

    # Video source
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: File '{args.video}' not found.")
            exit(1)
            
        cap = cv2.VideoCapture(args.video)
        print(f"Running on pre-recorded video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Running on live webcam.")
    
    # FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0 or np.isnan(fps):
        fps = 30    # in case no fps is defined

    print(f"Video FPS: {fps}")

    # Video writer
    video_writer = None
    if args.save:
        if args.video:
            video_name, ext_name = os.path.splitext(os.path.basename(args.video))
            save_path = os.path.join(args.save, video_name + "_Analysed.mp4")
        else:
            save_path = os.path.join(args.save, "Analysed.mp4")

        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # Initialise plot
    if args.plot:
        fig, axes = init_plot()

    # Initialise history variables
    measurements, predictions = {}, {}
    for point in POI:
        measurements[point] = np.empty(shape=(0,4))
        predictions[point] = np.empty(shape=(0,4))

    # Initialise juggle counter
    if args.events:
        juggle_counter = EventListCounter(args.events)
    else:
        juggle_counter = JuggleCounter(
            prominence=0.02,
            min_gap_frames=5,   # 0.16s @ 30fps
            max_distance=0.3,
            min_history_length=10,
            candidate_parts=["Left_Foot", "Right_Foot"] if args.feet_only else None,
            mode=('spread' if args.foot_spread else ('foot' if args.foot_peaks else 'ball')),
            foot_prominence=args.foot_prominence,
            foot_validation_window=args.foot_window,
            spread_prominence=args.spread_prominence,
        )

    # Full-video tracking for --save-plot
    if args.save_plot:
        ball_y_meas, ball_y_pred = [], []
        juggle_events = []          # list of (frame_idx, body_part)
        prev_counts = {}

    # Initialise visualiser
    visualiser = Visualiser(
        ball_colour=(0, 255, 0),        # Green
        landmark_colour=(0, 0, 255),    # Red
        text_colour=(0, 0, 0),          # Black
        text_thickness=3,
        bbox_thickness=2,
        landmark_radius=6
    )

    frame_idx = 0

    # Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or camera disconnected.")
            break

        # detect POI
        POIs = get_POI(frame, ball_conf=args.ball_conf, track=args.track)

        # update measurement history
        measurements = update_measurements(measurements, POIs)
        
        # update and predict
        predictions = predict_KF(measurements, predictions)
        #predictions= predict_para(measurements, predictions)

        # update plot
        if args.plot:
            update_plot(axes, measurements, predictions)

        # count juggle
        if isinstance(juggle_counter, EventListCounter):
            count = juggle_counter.update(frame_idx)
        else:
            count = juggle_counter.update(predictions)

        # log per-frame ball Y + juggle events for --save-plot
        if args.save_plot:
            ball_y_meas.append(measurements['Ball'][-1, 1])
            ball_y_pred.append(predictions['Ball'][-1, 1])
            for part, n in count.items():
                if n > prev_counts.get(part, 0):
                    juggle_events.append((frame_idx, part))
            prev_counts = dict(count)

        # draw on image
        visualiser.draw(frame, POIs, count)

        # Optional: draw total count and FPS
        visualiser.draw_total_count(frame, count)
        visualiser.draw_fps(frame, fps)

        frame_idx += 1

        # show image
        if not args.headless:
            cv2.imshow("Football Juggle Counter", frame)

        # write video
        if video_writer:
            video_writer.write(frame)

        # exit with 'q' (only meaningful when a window is up)
        if not args.headless:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # cleanup
    if video_writer:
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()

    # write full-video Y-position plot
    if args.save_plot:
        part_colours = {
            'Left_Foot':  '#1f77b4',
            'Right_Foot': '#ff7f0e',
            'Left_Knee':  '#2ca02c',
            'Right_Knee': '#d62728',
            'Head':       '#9467bd',
        }
        width = min(28, max(14, len(ball_y_meas) / 250))
        fig2, ax = plt.subplots(figsize=(width, 5))
        ax.plot(ball_y_meas, color='steelblue', linewidth=0.6, alpha=0.6, label='Measurement (YOLO)')
        ax.plot(ball_y_pred, color='crimson',   linewidth=0.8, label='Prediction (Kalman)')
        events_by_part = {}
        for f, part in juggle_events:
            events_by_part.setdefault(part, []).append(f)
        for part, frames in events_by_part.items():
            ax.scatter(frames, [0.04] * len(frames), marker='|', s=80, linewidths=1.2,
                       color=part_colours.get(part, 'gray'), alpha=0.9,
                       label=f'{part} ({len(frames)})')
        ax.set_ylim(1, 0)   # y=0 is top of frame
        ax.set_xlabel('Frame')
        ax.set_ylabel('Ball Y (normalised, 0=top, 1=bottom)')
        total = sum(count.values()) if count else 0
        ax.set_title(f'Ball Y-position over {len(ball_y_meas)} frames — {total} juggles')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=120)
        print(f'Saved Y-position plot to {args.save_plot}')

if __name__ == "__main__":
    main()