# CivicCam - AI-Powered Littering Detection System

An intelligent surveillance system that detects littering incidents in real-time using computer vision and machine learning. Built with YOLOv8 for object detection, EasyOCR for license plate recognition, and Streamlit for the web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Object Detection** - Detects waste, vehicles, license plates, and people using YOLOv8
- **Face Detection** - Identifies suspects in littering incidents
- **License Plate OCR** - Extracts plate text using EasyOCR
- **Telegram Alerts** - Automatic notifications when littering is detected
- **Multiple Input Sources** - Webcam, uploaded images/videos, RTSP streams
- **Dashboard** - Analytics and incident management interface
- **Incident Logging** - SQLite database for storing evidence

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for real-time detection)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/salahudheenthajudheen/civiccam.git
cd civiccam
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv civiccam_env
.\civiccam_env\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv civiccam_env
source civiccam_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your Telegram credentials
```

### Step 5: Download/Train Model
Place your trained model in `models/civiccam_best.pt` or train one using:
```bash
python scripts/train_model.py
```

## Usage

### Web Dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### Command Line Detection
```bash
# Detect in image
python detect.py --source image.jpg --save

# Detect in video
python detect.py --source video.mp4 --save

# Webcam with face detection
python detect.py --source 0 --show --face

# Full detection (OCR + events + face)
python detect.py --source image.jpg --ocr --events --face --save
```

## Telegram Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Get your Chat ID from [@userinfobot](https://t.me/userinfobot)
3. Add credentials to your `.env` file:
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Project Structure

```
civiccam/
├── app.py                 # Streamlit web dashboard
├── detect.py              # CLI detection script
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── models/
│   └── civiccam_best.pt   # Trained YOLO model
└── scripts/
    ├── detector.py        # Detection engine
    ├── ocr_engine.py      # License plate OCR
    ├── face_detector.py   # Face detection
    ├── event_detector.py  # Littering event logic
    ├── evidence_handler.py # Incident storage
    └── telegram_bot.py    # Telegram notifications
```

## Configuration

Edit `config.py` to customize:
- `DETECTION_CONF` - Detection confidence threshold (default: 0.15)
- `LITTERING_PROXIMITY_THRESHOLD` - Distance for event detection

Environment variables (in `.env`):
- `TELEGRAM_BOT_TOKEN` - Your Telegram bot token
- `TELEGRAM_CHAT_ID` - Your Telegram chat/group ID

## Model Classes

| Class | Description |
|-------|-------------|
| `license_plate` | Vehicle number plates |
| `waste` | Litter/garbage |
| `object` | Items being thrown |
| `public` | Public areas |

## Performance

- mAP50: 55.1%
- Inference: ~60ms per frame (GPU)
- Real-time capable on modern hardware

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Streamlit](https://streamlit.io/)
- [python-telegram-bot](https://python-telegram-bot.org/)
