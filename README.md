# CV — real-time face detection with OpenCV

Opens the MacBook's built-in camera, detects faces with a Haar cascade, and draws a green box around each one in a mirrored live view.

## Requirements

- macOS (camera lookup uses AVFoundation via PyObjC)
- Python 3

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Press `q` to quit.

## Notes

On macOS, OpenCV orders cameras by their unique ID, so an iPhone using Continuity Camera can take index 0. `find_camera_index("MacBook")` looks up the built-in camera by name instead of assuming index 0.
