"""
CivicCam Configuration
Central configuration for the littering detection system
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
INCIDENTS_DIR = BASE_DIR / "incidents"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Create directories
INCIDENTS_DIR.mkdir(exist_ok=True)
(INCIDENTS_DIR / "images").mkdir(exist_ok=True)
(INCIDENTS_DIR / "clips").mkdir(exist_ok=True)

# Model configuration
MODEL_PATH = MODELS_DIR / "civiccam_best.pt"

# Detection thresholds
DETECTION_CONF = 0.15  # Minimum confidence for detection (lowered to catch angled plates)
IOU_THRESHOLD = 0.45   # IoU threshold for NMS

# Class names (must match training)
CLASS_NAMES = ["license_plate", "object", "public", "waste"]
CLASS_COLORS = {
    "license_plate": (0, 255, 0),    # Green
    "object": (255, 165, 0),         # Orange
    "public": (0, 191, 255),         # Deep Sky Blue
    "waste": (255, 0, 0),            # Red
    "face": (255, 0, 255),           # Magenta (for face detection)
}

# Face detection settings
ENABLE_FACE_DETECTION = True
FACE_CONF_THRESHOLD = 0.5

# Littering event detection
LITTERING_PROXIMITY_THRESHOLD = 200  # Max pixels between waste and vehicle/person
LITTERING_TIME_WINDOW = 5.0          # Seconds to track for littering event

# Telegram Bot Configuration
# Load from environment variables (create .env file from .env.example)
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Streamlit config fallback
try:
    import streamlit as st
    if "TELEGRAM_BOT_TOKEN" in st.secrets: TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    if "TELEGRAM_CHAT_ID" in st.secrets: TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    if "SUPABASE_URL" in st.secrets: SUPABASE_URL = st.secrets["SUPABASE_URL"]
    if "SUPABASE_KEY" in st.secrets: SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    pass

# Database (Local Backup Config)
DATABASE_PATH = INCIDENTS_DIR / "incidents.db"

# Video/Image settings
MAX_IMAGE_SIZE = 1280
VIDEO_FPS = 30

# Alert settings
ALERT_COOLDOWN = 30  # Seconds between alerts for same vehicle
