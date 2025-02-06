# Chef Hat Detection System

A computer vision system that detects whether kitchen staff are wearing chef hats, using YOLO object detection and custom training.

## Features

- Real-time chef hat detection in videos
- Custom training support with LoRA
- Detailed analytics and reporting
- Support for multiple hat colors and styles
- Performance metrics visualization

## Setup

1. Install dependencies:
```bash
pip install ultralytics torch opencv-python numpy pillow scikit-learn matplotlib seaborn
```

2. Prepare your dataset:
- Place images in `dataset/images/`
- Create annotations in `dataset/annotations.json`

3. Run the program:
```bash
python chef_hat_detector.py
```

## Usage

The system supports two modes:

1. Training Mode:
   - Train the model on your custom dataset
   - Fine-tune using LoRA
   - Monitor training metrics

2. Video Analysis Mode:
   - Analyze videos for chef hat compliance
   - Generate detailed reports
   - Visualize detection results

## Project Structure

- `chef_hat_detector.py`: Main program file
- `dataset/`: Training data directory
- `videos/`: Input videos directory
- `results/`: Output directory for analysis results

## License

MIT License 