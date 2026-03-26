# 🚔 CivicCam - AI Littering Detection System

An AI-powered system that detects littering incidents by identifying **waste**, **license plates**, and **suspect faces** simultaneously, then sends alerts via Telegram.

## 🎯 Features

- **Real-time Detection**: Detects waste, vehicles, and people using YOLOv8
- **License Plate OCR**: Reads vehicle registration numbers
- **Face Detection**: Captures suspect faces for identification
- **Smart Alerts**: Only triggers when ALL THREE are detected together
- **Telegram Integration**: Sends instant alerts with evidence images
- **Web Dashboard**: Monitor incidents via Streamlit interface
- **Supabase Backend**: Cloud database & image storage

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ 
- macOS / Linux / Windows
- Webcam (for live detection)

### 1. Clone & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd civiccam-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

**Telegram Setup:**
1. Open Telegram, search for `@BotFather`
2. Send `/newbot` and follow instructions to get your token
3. Add the bot to a group/channel
4. Get chat ID from `@userinfobot` or `@getidsbot`

**Supabase Setup:**
1. Create a project at [supabase.com](https://supabase.com)
2. Run `scripts/schema.sql` in the SQL Editor
3. Copy the URL and anon key from Project Settings > API

### 3. Run the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run the web dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ☁️ Deployment (Streamlit Cloud + Supabase)

### Deploy to Streamlit Cloud

1. **Push to GitHub** — Make sure your repo is on GitHub
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Connect your repo** — Select the repository and `app.py` as the main file
4. **Add Secrets** — In the Streamlit Cloud dashboard, go to **Settings > Secrets** and add:
   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-anon-key"
   TELEGRAM_BOT_TOKEN = "your-bot-token"
   TELEGRAM_CHAT_ID = "your-chat-id"
   ```
5. **Deploy** — Click Deploy and wait for the app to build

### Supabase Backend Setup

Run the SQL schema in your Supabase SQL Editor:

```sql
-- See scripts/schema.sql for the full schema
-- Creates: incidents table, storage bucket, RLS policies, indexes
```

### Important Files for Deployment

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `packages.txt` | System-level apt packages (for OpenCV) |
| `.streamlit/secrets.toml` | Local secrets (NOT committed to git) |
| `scripts/schema.sql` | Supabase database schema |

---

## 📁 Project Structure

```
civiccam-model/
├── app.py                 # Main Streamlit web app
├── detect.py              # CLI detection script
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── packages.txt           # System packages for Streamlit Cloud
├── .env.example           # Environment variable template
├── .streamlit/
│   ├── secrets.toml       # Local secrets (gitignored)
│   └── secrets.toml.example  # Secrets template
├── models/
│   └── civiccam_best.pt   # Trained YOLOv8 model
├── scripts/
│   ├── detector.py        # Object detection module
│   ├── ocr_engine.py      # License plate OCR
│   ├── face_detector.py   # Face detection
│   ├── telegram_bot.py    # Telegram alerts
│   ├── event_detector.py  # Littering event logic
│   ├── evidence_handler.py # Supabase + SQLite backend
│   └── schema.sql         # Supabase database schema
└── datasets/              # Training datasets
```

---

## 🎮 Usage

### Web Dashboard (Recommended)

```bash
streamlit run app.py
```

### Command Line

```bash
# Detect from webcam
python detect.py --source 0 --show

# Detect from image
python detect.py --source path/to/image.jpg --show

# Detect from video
python detect.py --source path/to/video.mp4 --save

# Enable all features
python detect.py --source 0 --ocr --face --events --show
```

---

## 🧠 How Detection Works

The system requires **waste + suspect** to trigger an alert:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│    Waste    │ + │License Plate│ + │    Face     │ = 🚨 ALERT
│  Detected   │   │  Detected   │   │  Detected   │
└─────────────┘   └─────────────┘   └─────────────┘
```

When triggered, the system:
1. Captures the full scene
2. Crops the suspect's face
3. Crops the license plate
4. Reads the plate number with OCR
5. Saves evidence to Supabase database
6. Uploads images to Supabase Storage
7. Sends Telegram alert with all images

---

## 📊 Model Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `license_plate` | Vehicle license plates |
| 1 | `object` | Generic objects |
| 2 | `public` | People/pedestrians |
| 3 | `waste` | Litter/garbage |
| - | `face` | Detected by separate face model |

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
DETECTION_CONF = 0.35      # Confidence threshold
FACE_CONF_THRESHOLD = 0.5  # Face detection threshold
```

---

## 📱 Telegram Alert Format

When a littering incident is detected:

```
🚨 LITTERING DETECTED

🚗 Plate: KL01AB1234
👤 Suspect: Face captured
🗑️ Evidence: Waste detected

📊 Confidence Scores:
• License Plate: 94%
• Face Detection: 87%
• Waste Detection: 82%

🕐 Time: 27/01/2026 • 03:10:45
🆔 Case: #42
```

Plus 4 evidence images: scene, face, plate, waste.

---

## 🏋️ Training Your Own Model

1. Prepare dataset in YOLO format
2. Upload `CivicCam_Training.ipynb` to Google Colab
3. Follow the notebook instructions
4. Replace `models/civiccam_best.pt` with your trained model

---

## 🛠️ Tech Stack

- **YOLOv8** — Object detection
- **EasyOCR** — License plate reading
- **Streamlit** — Web interface
- **Supabase** — Cloud database & image storage
- **Telegram Bot API** — Real-time alerts
- **OpenCV** — Image processing

---

## 📊 Model Performance

- mAP50: 92.3%
- Precision: 95.8%
- Recall: 87.7%

---

## 📝 License

MIT License - Feel free to use and modify for your projects.

---

## 🤝 Contributing

Pull requests welcome! Please open an issue first to discuss changes.
