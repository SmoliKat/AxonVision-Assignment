"""Three-process motion-detection pipeline: Streamer -> Detector -> Viewer."""
import multiprocessing as mp
import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np


EOS = None  # end-of-stream sentinel used on every queue
QUEUE_MAXSIZE = 2  # bounded queues give us backpressure: Streamer blocks when pipeline is full


def streamer(video_path: str, frame_queue: mp.Queue) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Signal EOS so downstream doesn't wait for frames that will never arrive.
        frame_queue.put(EOS)
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_queue.put((frame_id, frame))
            frame_id += 1
    finally:
        cap.release()
        frame_queue.put(EOS)


def detector(frame_queue: mp.Queue, result_queue: mp.Queue) -> None:
    prev_gray: Optional[np.ndarray] = None
    try:
        while True:
            msg = frame_queue.get()
            if msg is EOS:
                return
            frame_id, frame = msg

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections: list = []
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # imutils.grab_contours equivalent: OpenCV 3 returns 3 items, OpenCV 4 returns 2.
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                detections = [cv2.boundingRect(c) for c in cnts]
            prev_gray = gray

            result_queue.put((frame_id, frame, detections))
    finally:
        # Guarantee downstream unblocks even if this process exits by exception.
        result_queue.put(EOS)


def viewer(result_queue: mp.Queue, fps: float) -> None:
    frame_interval = 1.0 / fps
    start_time: Optional[float] = None
    window = "Pipeline"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            msg = result_queue.get()
            if msg is EOS:
                break
            frame_id, frame, detections = msg

            # Only the Viewer mutates pixels; Streamer/Detector never touch the image.
            # Blur first (in-place on the ROI), then draw rectangles on top so they stay crisp.
            for x, y, w, h in detections:
                frame[y:y + h, x:x + w] = cv2.GaussianBlur(frame[y:y + h, x:x + w], (21, 21), 0)
            for x, y, w, h in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Presentation-time pacing: absorb jitter without drifting.
            if start_time is None:
                start_time = time.perf_counter()
            sleep_for = (start_time + frame_id * frame_interval) - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

            cv2.imshow(window, frame)
            cv2.waitKey(1)  # pump GUI events; pacing is done by time.sleep above.
    finally:
        cv2.destroyAllWindows()


def probe_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 else "People - 6387.mp4"
    fps = probe_fps(video_path)

    frame_queue: mp.Queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
    result_queue: mp.Queue = mp.Queue(maxsize=QUEUE_MAXSIZE)

    procs = [
        mp.Process(target=streamer, args=(video_path, frame_queue), name="Streamer"),
        mp.Process(target=detector, args=(frame_queue, result_queue), name="Detector"),
        mp.Process(target=viewer, args=(result_queue, fps), name="Viewer"),
    ]
    for p in procs:
        p.start()

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        # Ctrl-C on the main process: ask children to die, then wait for them.
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
