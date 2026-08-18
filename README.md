# AxonVision Assignment - Motion Detection Pipeline

A three-process motion-detection pipeline built for the Axon Vision home assignment.

```
Streamer  ->  Detector  ->  Viewer
```

- **Streamer** reads frames from a video file and pushes them onto a queue.
- **Detector** runs the supplied `basic_vmd.py` algorithm and forwards the frame plus bounding-box detections. Does not draw.
- **Viewer** blurs each detected region, draws green rectangles, stamps the current wall-clock time in the top-left, and displays the video at the source FPS. The only component that touches pixels.

IPC is two bounded `multiprocessing.Queue`s (`maxsize=2`); end-of-stream is a `None` sentinel that propagates through the pipeline.

## Requirements

- Python 3.8+
- `opencv-python`
- `numpy` (installed as a dependency of `opencv-python`)
- A display (native window on Windows/macOS, or X server on Linux). The Viewer uses `cv2.imshow`; **headless environments will not work.**

Install:

```
python -m pip install --upgrade pip
python -m pip install opencv-python numpy
```

## How to run

From the repository root:

```
python pipeline.py
```

Uses the bundled `People - 6387.mp4` by default. To use a different file:

```
python pipeline.py path/to/other/video.mp4
```

Expected behavior:
- An OpenCV window titled **Pipeline** opens.
- Video plays at close to the source's real duration.
- Moving regions are Gaussian-blurred inside green rectangles.
- Current wall-clock time (`HH:MM:SS`) is drawn in the top-left corner.
- When the video ends, all three processes exit and the shell prompt returns.

Ctrl-C in the terminal terminates all child processes cleanly.

## Git tags — reviewing the assignment in stages

The assignment was implemented in three tagged stages so each milestone can be reviewed independently:

| Tag | What was added |
| --- | -------------- |
| `stage-a` | Three-process pipeline; rectangles + timestamp in Viewer; smooth presentation-time playback; sentinel-based shutdown on natural EOF. |
| `stage-b` | Gaussian blur of detected ROIs added in Viewer (blur first, rectangle on top). No changes elsewhere. |
| `stage-c` | Shutdown hardening: `try/finally` in Detector and Viewer, `KeyboardInterrupt` handling in `main`. |

Check out any stage locally:

```
git checkout stage-a         # see just Stage A
git checkout stage-b         # see Stage A + B
git checkout stage-c         # see Stage A + B + C
git checkout main            # return to the latest state
```

Or browse each stage on GitHub:

- https://github.com/SmoliKat/AxonVision-Assignment/tree/stage-a
- https://github.com/SmoliKat/AxonVision-Assignment/tree/stage-b
- https://github.com/SmoliKat/AxonVision-Assignment/tree/stage-c

## Repository layout

```
.
├── Assignment.pdf     # The assignment brief (Hebrew).
├── basic_vmd.py       # The supplied motion-detection algorithm.
├── People - 6387.mp4  # Sample video (1280x720, 25 fps, ~14 s).
├── pipeline.py        # The solution: 3-process pipeline.
└── README.md          # This file.
```
