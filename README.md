# Center Frame

A real-time face tracking application that automatically zooms and pans to keep the subject centered in the video feed. It uses YOLOv8 for face detection and applies smoothing to camera movements.

<video width="320" height="240">
  <source src="center-frame-2025-12-03-17-57-33.mp4" type="video/mp4">
</video>

## Usage

Run the application:

```bash
uv run main.py
```

### Recording

To start recording after a delay (e.g., 5 seconds), use the `--record` flag:

```bash
uv run main.py --record 5
```

Video is saved as `center-frame-<timestamp>.mp4`.

### Don't have uv installed?
1. Install uv by following this guide: [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/) (Recommended)

**OR**

2. Use pip:
    
```bash
python -m venv env
pip install -r requirements.txt
python main.py
# or
python main.py --record 5 # to enable recording
```

### Controls

- Press **'q'** to quit.

## Tech Stack

- **Python**
- **OpenCV** (Video processing)
- **YOLOv8** (Face detection)
- **Supervision** (Detections handling)
- **PyTorch** (Inference)
- **Hugging Face Hub** (Model downloading)

## Credits

- Face Detection Model: [YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection) by Arnab Dhar.

## License

This project is licensed under the MIT License.
