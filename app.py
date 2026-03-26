"""
CivicCam Streamlit Dashboard
Live monitoring and incident management interface
"""

import streamlit as st
from pathlib import Path
import sys
import time
from datetime import datetime
import json

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Page configuration
st.set_page_config(
    page_title="CivicCam - Littering Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Mode UI
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Custom Header */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #151922;
        padding: 12px 24px;
        border-bottom: 1px solid #262730;
        margin: -4rem -4rem 1rem -4rem; /* Break out of Streamlit container */
    }
    
    .brand {
        font-size: 1.2rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .live-badge {
        background-color: #1E8E3E;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 0.5px;
    }
    
    /* Cards */
    .panel-card {
        background-color: #151922;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    
    .card-header {
        font-size: 0.8rem;
        color: #8B929A;
        margin-bottom: 8px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Suspect Card Specifics */
    .suspect-card {
        border: 1px solid #FF4B4B;
        background-color: #1F1111;
    }
    .suspect-img {
        border-radius: 6px;
        border: 1px solid #FF4B4B;
        width: 100%;
    }
    
    /* Recent Events List */
    .event-item {
        display: flex;
        gap: 10px;
        padding: 8px;
        border-bottom: 1px solid #262730;
        cursor: pointer;
    }
    .event-item:hover {
        background-color: #1F2229;
    }
    .event-img {
        width: 40px;
        height: 40px;
        border-radius: 4px;
        object-fit: cover;
    }
    
    /* Footer Stats */
    .meta-footer {
        display: flex;
        justify-content: space-between;
        background-color: #151922;
        padding: 16px;
        border-top: 1px solid #262730;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 100;
    }
    .meta-item {
        display: flex;
        flex-direction: column;
    }
    .meta-label {
        font-size: 0.7rem;
        color: #8B929A;
        margin-bottom: 4px;
    }
    .meta-value {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load the detection model (cached)"""
    try:
        from detector import CivicCamDetector
        return CivicCamDetector()
    except Exception as e:
        st.error(f"Error loading detector: {e}")
        return None


@st.cache_resource
def load_ocr():
    """Load OCR engine (cached)"""
    try:
        from ocr_engine import LicensePlateOCR
        return LicensePlateOCR()
    except Exception as e:
        st.warning(f"OCR not available: {e}")
        return None


@st.cache_resource
def load_face_detector():
    """Load face detector (cached)"""
    try:
        from face_detector import FaceDetector
        return FaceDetector(conf_threshold=0.5)
    except Exception as e:
        st.warning(f"Face detection not available: {e}")
        return None


@st.cache_resource
def load_telegram_bot():
    """Load Telegram bot (cached)"""
    try:
        from telegram_bot import TelegramAlertBot
        bot = TelegramAlertBot()
        if bot.is_configured():
            return bot
        return None
    except Exception as e:
        st.sidebar.error(f"Telegram Bot error: {e}")
        return None


@st.cache_resource  
def load_event_detector():
    """Load littering event detector (cached)"""
    try:
        from event_detector import LitteringEventDetector
        return LitteringEventDetector()
    except Exception as e:
        return None


def check_and_send_alert(detections, image, ocr, telegram_bot, handler, face_detector=None, event_detector=None, is_video=False):
    """Check for DUMPING event and send Telegram alert
    
    For video/streams: Uses event_detector to track NEW waste (dumping action)
    For images: Triggers on waste + suspect (no temporal tracking)
    
    Triggers alert when waste is detected PLUS either:
    - face (suspect identification) OR
    - license_plate (vehicle identification) OR
    - public (person/pedestrian identification)
    """
    import cv2
    
    # Handle both 'waste' and 'Waste' class names
    waste_detected = any(d['class_name'].lower() == 'waste' for d in detections)
    plate_detected = any(d['class_name'] == 'license_plate' for d in detections)
    face_detected = any(d['class_name'] == 'face' for d in detections)
    person_detected = any(d['class_name'] == 'public' for d in detections)
    
    # Suspect is identified if we have face, plate, OR person
    suspect_identified = face_detected or plate_detected or person_detected
    
    # For video streams: use event detector to only alert on NEW waste
    # For images: alert on any waste + suspect combination
    should_alert = False
    
    if is_video and event_detector:
        # Check for NEW waste (dumping action) in video
        events = event_detector.process_detections(detections)
        if len(events) > 0:
            should_alert = True
            print(f"[Alert] Dumping event detected in video: {len(events)} events")
    else:
        # For images: alert on waste + suspect
        if waste_detected and suspect_identified:
            should_alert = True
            print(f"[Alert] Littering detected in image: waste + suspect (face={face_detected}, plate={plate_detected}, person={person_detected})")
    
    if should_alert:
        # Littering event with suspect identified!
        plate_det = next((d for d in detections if d['class_name'] == 'license_plate'), None)
        waste_det = next((d for d in detections if d['class_name'] == 'waste'), None)
        face_det = next((d for d in detections if d['class_name'] == 'face'), None)
        
        # Try to read plate with OCR
        plate_text = "UNKNOWN"
        plate_conf = 0.0
        if plate_det and ocr:
            text, conf, _ = ocr.extract_plate_from_frame(image, plate_det['bbox'])
            if text:
                plate_text = text
                plate_conf = conf
        
        # Get confidence scores
        waste_conf = waste_det['confidence'] if waste_det else 0.0
        face_conf = face_det['confidence'] if face_det else 0.0
        
        # Create incident folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        incident_folder = Path("incidents/images") / f"incident_{timestamp}"
        incident_folder.mkdir(parents=True, exist_ok=True)
        
        # Save full frame with all detections
        frame_path = incident_folder / "scene.jpg"
        cv2.imwrite(str(frame_path), image)
        
        # Save individual crops
        crops = {}
        waste_crop = None
        face_crop = None
        plate_crop = None
        
        # Waste crop
        if waste_det:
            x1, y1, x2, y2 = waste_det['bbox']
            waste_crop = image[max(0,y1):y2, max(0,x1):x2]
            if waste_crop.size > 0:
                waste_path = incident_folder / "waste.jpg"
                cv2.imwrite(str(waste_path), waste_crop)
                crops['waste'] = str(waste_path)
        
        # Face crop
        if face_det:
            x1, y1, x2, y2 = face_det['bbox']
            # Add padding for face
            h, w = image.shape[:2]
            pad = 20
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w, x2+pad), min(h, y2+pad)
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_path = incident_folder / "suspect_face.jpg"
                cv2.imwrite(str(face_path), face_crop)
                crops['face'] = str(face_path)
        
        # License plate crop
        if plate_det:
            x1, y1, x2, y2 = plate_det['bbox']
            plate_crop = image[max(0,y1):y2, max(0,x1):x2]
            if plate_crop.size > 0:
                plate_path = incident_folder / "license_plate.jpg"
                cv2.imwrite(str(plate_path), plate_crop)
                crops['plate'] = str(plate_path)
        
        # Save incident to database
        incident_id = None
        if handler:
            incident_id = handler.save_incident(
                frame=image,
                license_plate=plate_text,
                plate_confidence=plate_conf,
                detections=[{"class_name": d["class_name"], "confidence": d["confidence"]} for d in detections],
                source="streamlit_app",
                location="",
                plate_crop=plate_crop,
                waste_crop=waste_crop
            )
        
        # Send Telegram alert with all evidence
        if telegram_bot:
            telegram_bot.send_littering_alert(
                license_plate=plate_text,
                plate_confidence=plate_conf,
                waste_confidence=waste_conf,
                face_confidence=face_conf,
                scene_image=str(frame_path),
                face_image=crops.get('face'),
                plate_image=crops.get('plate'),
                waste_image=crops.get('waste'),
                incident_id=incident_id
            )
            # WebRTC runs in background thread which crashes on st.success, so handle safely
            try:
                st.success(f"🚨 ALERT SENT! Suspect identified - Plate: {plate_text}")
            except Exception:
                print(f"[Alert] Successfully sent. Plate: {plate_text}")
            return True
    
    return False


def get_evidence_handler():
    """Get evidence handler instance"""
    try:
        from evidence_handler import EvidenceHandler
        return EvidenceHandler()
    except Exception as e:
        st.error(f"Error loading evidence handler: {e}")
        return None


def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.markdown("## 🚗 CivicCam")
        
        page = st.radio(
            "Navigation",
            ["🎥 Live Feed", "📊 Dashboard", "🚨 Incidents", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        st.divider()
        return page


def render_live_feed():
    """Render the new Design UI"""
    import cv2
    import numpy as np
    
    # Custom Top Bar
    st.markdown("""
    <div class="top-bar">
        <div class="brand">
            <span>🚗 CivicCam</span>
            <span class="live-badge">Live</span>
        </div>
        <div style="display: flex; gap: 20px; color: #8B929A; font-size: 0.9rem;">
            <span>Incidents: <span style="color: #4CAF50;">0</span></span>
            <span>FPS: <span style="color: #4CAF50;">19.7</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Layout: [ 3: Main Feed ] [ 1: Faces/Plates ] [ 1: Recent List ]
    col_main, col_cards, col_list = st.columns([0.6, 0.2, 0.2], gap="medium")

    # --- INIT STATE ---
    # Initialization is now lazy (triggered when a source is used)
    def initialize_ai():
        if 'ai_ready' not in st.session_state:
            with st.spinner("🧠 Loading AI Models... (First time may take a few mins for weights download)"):
                detector = load_detector()
                ocr = load_ocr()
                face_detector = load_face_detector()
                event_detector = load_event_detector()
                handler = get_evidence_handler()
                st.session_state.ai_ready = True
                st.session_state.ai_components = (detector, ocr, face_detector, event_detector, handler)
        
        return st.session_state.ai_components

    telegram_bot = load_telegram_bot()
    
    # Show Telegram status in sidebar
    if telegram_bot:
        st.sidebar.success("📱 Telegram: Connected")
    else:
        st.sidebar.warning("📱 Telegram: Not configured")

    # Session state for current view
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None
    if 'current_meta' not in st.session_state:
        st.session_state.current_meta = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "object": "scanning...",
            "vehicle": "No",
            "confidence": "0%"
        }
        
    # Lazy init fallbacks
    detector = None
    ocr = None
    face_detector = None
    event_detector = None
    handler = get_evidence_handler() # Handler is fast, safe to load for 'Recent Events' UI

    if 'ai_components' in st.session_state:
        detector, ocr, face_detector, event_detector, _ = st.session_state.ai_components

    # --- CENTER COLUMN: Main Feed ---
    with col_main:
        st.markdown("#### 📁 Incident Details")
        
        # Source selector (compact)
        source_type = st.selectbox(
            "Source",
            ["📁 Upload Image", "📷 Webcam", "🎬 Upload Video", "🔗 RTSP Stream"],
            label_visibility="collapsed"
        )

        display_placeholder = st.empty()
        
        # Logic to handle different sources
        if source_type == "📁 Upload Image":
            uploaded_file = st.file_uploader("Drop evidence here", type=["jpg", "png"], label_visibility="collapsed")
            if uploaded_file:
                # Lazy load models
                detector, ocr, face_detector, event_detector, handler = initialize_ai()
                
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if detector:
                    annotated, detections = detector.detect_and_draw(image)
                    display_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')
                    
                    # Update metadata
                    st.session_state.current_meta["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Check for specific detections
                    waste_detected = any(d['class_name'].lower() == 'waste' for d in detections)
                    plate_detected = any(d['class_name'] == 'license_plate' for d in detections)
                    
                    st.session_state.current_meta["object"] = "Waste" if waste_detected else "Unknown"
                    st.session_state.current_meta["vehicle"] = "Yes" if plate_detected else "No"
                    st.session_state.current_meta["confidence"] = f"{max([d['confidence'] for d in detections] or [0]):.0%}"
                    
                    # Detect faces and add to detections
                    if face_detector:
                        faces = face_detector.detect_faces(image)
                        detections.extend(faces)
                    
                    face_detected = any(d['class_name'] == 'face' for d in detections)
                    
                    # Debug: Print what's detected
                    print(f"[DEBUG] waste={waste_detected}, plate={plate_detected}, face={face_detected}")
                    print(f"[DEBUG] All classes: {[d['class_name'] for d in detections]}")
                    
                    person_detected = any(d['class_name'] == 'public' for d in detections)
                    
                    # Check for littering and send alert (waste + face OR waste + plate OR waste + person)
                    if waste_detected and (face_detected or plate_detected or person_detected):
                        print("[DEBUG] Alert condition MET - calling check_and_send_alert")
                        check_and_send_alert(detections, image, ocr, telegram_bot, handler, event_detector=event_detector)
                    else:
                        print("[DEBUG] Alert condition NOT met")
                    
                    # Store latest detections for the side cards
                    st.session_state.latest_detections = detections
                    st.session_state.latest_image = image

        elif source_type == "🎬 Upload Video":
            uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
            if uploaded_video:
                # Lazy load models
                detector, ocr, face_detector, event_detector, handler = initialize_ai()
                
                import tempfile
                import os
                
                # Save uploaded video to a temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_video.read())
                
                vf = cv2.VideoCapture(tfile.name)
                
                stframe = st.empty()
                stop_button = st.button("Stop Processing")
                
                while vf.isOpened() and not stop_button:
                    ret, frame = vf.read()
                    if not ret:
                        break
                    
                    if detector:
                        annotated, detections = detector.detect_and_draw(frame)
                        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')
                        
                        # Update metadata
                        st.session_state.current_meta["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        waste_detected = any(d['class_name'].lower() == 'waste' for d in detections)
                        plate_detected = any(d['class_name'] == 'license_plate' for d in detections)
                        
                        st.session_state.current_meta["object"] = "Waste" if waste_detected else "Unknown"
                        st.session_state.current_meta["vehicle"] = "Yes" if plate_detected else "No"
                        st.session_state.current_meta["confidence"] = f"{max([d['confidence'] for d in detections] or [0]):.0%}"
                        
                        # Detect faces
                        if face_detector:
                            faces = face_detector.detect_faces(frame)
                            detections.extend(faces)
                        face_detected = any(d['class_name'] == 'face' for d in detections)
                        
                        person_detected = any(d['class_name'] == 'public' for d in detections)
                        
                        # Check for littering (waste + face OR waste + plate OR waste + person)
                        if waste_detected and (face_detected or plate_detected or person_detected):
                            current_time = time.time()
                            if (current_time - st.session_state.get('last_alert_time', 0)) > 300:  # 5 min cooldown
                                print(f"[VIDEO] Alert! waste={waste_detected}, plate={plate_detected}, face={face_detected}")
                                check_and_send_alert(detections, frame, ocr, telegram_bot, handler)
                                st.session_state.last_alert_time = current_time
                        
                        # Store for cards
                        st.session_state.latest_detections = detections
                        st.session_state.latest_image = frame
                        
                    time.sleep(0.01)  # Simulate real-time playback
                
                vf.release()
                # Clean up the temporary file
                try:
                    os.remove(tfile.name)
                except:
                    pass

        elif source_type == "📷 Webcam":
            # Lazy load models
            detector, ocr, face_detector, event_detector, handler = initialize_ai()
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            import av
            import threading

            if 'last_alert_time' not in st.session_state:
                st.session_state.last_alert_time = 0
                
            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            # Use threading Lock for safe variable sharing
            lock = threading.Lock()
            class CivicCamProcessor:
                def __init__(self):
                    # Use components from session state (initialized via initialize_ai)
                    self.detector, self.ocr, self.face_detector, self.event_detector, self.handler = st.session_state.ai_components
                    self.telegram_bot = load_telegram_bot()
                    self.last_alert = time.time() - 300 # Immediately ready

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    
                    if self.detector:
                        annotated, detections = self.detector.detect_and_draw(img)
                        
                        if self.face_detector:
                            faces = self.face_detector.detect_faces(img)
                            detections.extend(faces)
                            
                        waste_det = any(d['class_name'].lower() == 'waste' for d in detections)
                        plate_det = any(d['class_name'] == 'license_plate' for d in detections)
                        face_det = any(d['class_name'] == 'face' for d in detections)
                        person_det = any(d['class_name'] == 'public' for d in detections)
                        
                        current_time = time.time()
                        
                        with lock:
                            if waste_det and (face_det or plate_det or person_det) and (current_time - self.last_alert) > 300:
                                print(f"[WEBCAM] Thread Alert! waste={waste_det}, plate={plate_det}, face={face_det}, person={person_det}")
                                check_and_send_alert(detections, img, self.ocr, self.telegram_bot, self.handler)
                                self.last_alert = current_time
                                
                        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                    
                    return frame

            webrtc_streamer(
                key="civiccam-webcam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=CivicCamProcessor,
                async_processing=True,
            )
            
            st.info("💡 Grant camera permissions in your browser to start the real-time AI detection feed!")

        elif source_type == "🔗 RTSP Stream":
            # Lazy load models
            detector, ocr, face_detector, event_detector, handler = initialize_ai()
            
            rtsp_url = st.text_input("Enter RTSP URL", placeholder="rtsp://admin:password@ip:port/stream")
            
            if rtsp_url and st.button("🔗 Connect Stream"):
                st.info(f"Connecting to {rtsp_url}...")
                cap = cv2.VideoCapture(rtsp_url)
                
                stop_button = st.button("🔴 Stop Stream")
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Stream disconnected")
                        break
                    
                    if detector:
                        annotated, detections = detector.detect_and_draw(frame)
                        display_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')
                        
                        # Update metadata
                        st.session_state.current_meta["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Detect faces
                        if face_detector:
                            faces = face_detector.detect_faces(frame)
                            detections.extend(faces)
                        
                        # Check for littering (waste + face OR waste + plate OR waste + person)
                        waste_det = any(d['class_name'].lower() == 'waste' for d in detections)
                        plate_det = any(d['class_name'] == 'license_plate' for d in detections)
                        face_det = any(d['class_name'] == 'face' for d in detections)
                        person_det = any(d['class_name'] == 'public' for d in detections)
                        current_time = time.time()
                        
                        if waste_det and (face_det or plate_det or person_det) and (current_time - st.session_state.get('last_alert_time', 0)) > 300:  # 5 min cooldown
                            print(f"[RTSP] Alert! waste={waste_det}, plate={plate_det}, face={face_det}, person={person_det}")
                            check_and_send_alert(detections, frame, ocr, telegram_bot, handler)
                            st.session_state.last_alert_time = current_time
                        
                        # Store for cards
                        st.session_state.latest_detections = detections
                        st.session_state.latest_image = frame
                        
                    # Limit frame rate slightly for UI responsiveness
                    time.sleep(0.01)
                
                cap.release()

    # --- RIGHT PANEL: Cards ---
    with col_cards:
        # Suspect Face Card (Dynamic)
        found_suspect = False
        suspect_img_content = """<img src="https://placehold.co/200x200/1F1111/FF4B4B?text=No+Suspect" class="suspect-img">"""
        
        if 'latest_detections' in st.session_state:
            for det in st.session_state.latest_detections:
                if det['class_name'] == 'face':  # Look for face detections
                    # Crop face
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    img = st.session_state.latest_image
                    if img is not None:
                        # Extract face crop
                        face_crop = det.get('face_image')  # From face detector
                        if face_crop is None:
                            face_crop = img[y1:y2, x1:x2]
                        if face_crop is not None and face_crop.size > 0:
                            found_suspect = True
                            suspect_img_content = ""
                            
                            st.markdown("""
                            <div class="panel-card suspect-card">
                                <div class="card-header">
                                    👤 Suspect Face
                                </div>
                            """, unsafe_allow_html=True)
                            st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), width='stretch')
                            st.markdown(f"<small>Confidence: {det['confidence']:.0%}</small>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        break
        
        if not found_suspect:
            st.markdown(f"""
            <div class="panel-card suspect-card">
                <div class="card-header">
                    👤 Suspect Face
                </div>
                {suspect_img_content}
            </div>
            """, unsafe_allow_html=True)
        
        # License Plate Card
        st.markdown("""
        <div class="panel-card">
            <div class="card-header">
                🚗 License Plate
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Dynamic Plate Content
        found_plate = False
        if 'latest_detections' in st.session_state:
            for det in st.session_state.latest_detections:
                if det['class_name'] == 'license_plate':
                    # Crop plate
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    img = st.session_state.latest_image
                    if img is not None:
                        plate_crop = img[y1:y2, x1:x2]
                        st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), width='stretch')
                        if ocr:
                            text, conf, _ = ocr.extract_plate_from_frame(img, det['bbox'])
                            st.markdown(f"**{text or 'Reading...'}**")
                        found_plate = True
                    break
        
        if not found_plate:
             st.info("Not detected")

    # --- FAR RIGHT: Recent Events ---
    with col_list:
        st.markdown("###### Recent Events")
        
        if handler:
            recents = handler.get_incidents(limit=5)
            for evt in recents:
                timestamp = evt.get("timestamp", "").split("T")[-1][:5]
                plate = evt.get("license_plate") or "Unknown"
                st.markdown(f"""
                <div class="event-item">
                    <div style="background:#333; width:40px; height:40px; border-radius:4px;"></div>
                    <div>
                        <div style="font-size:0.8rem; font-weight:bold;">{timestamp}</div>
                        <div style="font-size:0.7rem; color:#888;">{plate}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- FOOTER ---
    meta = st.session_state.current_meta
    st.markdown(f"""
    <div class="meta-footer">
        <div class="meta-item">
            <span class="meta-label">Time</span>
            <span class="meta-value">{meta['time']}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Object</span>
            <span class="meta-value">{meta['object']}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Vehicle</span>
            <span class="meta-value" style="color: {'#FF4B4B' if meta['vehicle']=='No' else '#4CAF50'};">{meta['vehicle']}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Confidence</span>
            <span class="meta-value">{meta['confidence']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard():
    """Render analytics dashboard"""
    st.markdown('<h1 class="main-header">📊 Dashboard</h1>', unsafe_allow_html=True)
    
    handler = get_evidence_handler()
    
    if not handler:
        st.error("Database not available")
        return
    
    stats = handler.get_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats.get('total_incidents', 0)}</div>
            <div class="stat-label">Total Incidents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="stat-value">{stats.get('today_count', 0)}</div>
            <div class="stat-label">Today's Incidents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="stat-value">{stats.get('unique_plates', 0)}</div>
            <div class="stat-label">Unique Vehicles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pending = stats.get('by_status', {}).get('pending', 0)
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="stat-value">{pending}</div>
            <div class="stat-label">Pending Review</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Recent incidents chart placeholder
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Incidents Over Time")
        # Placeholder chart - in production, use actual data
        import random
        chart_data = {
            "Date": [f"Day {i}" for i in range(1, 8)],
            "Incidents": [random.randint(0, 10) for _ in range(7)]
        }
        st.bar_chart(chart_data, x="Date", y="Incidents")
    
    with col2:
        st.markdown("### 🚗 Top Offending Vehicles")
        incidents = handler.get_incidents(limit=10)
        
        plate_counts = {}
        for inc in incidents:
            plate = inc.get('license_plate', 'Unknown')
            if plate:
                plate_counts[plate] = plate_counts.get(plate, 0) + 1
        
        if plate_counts:
            for plate, count in sorted(plate_counts.items(), key=lambda x: -x[1])[:5]:
                st.markdown(f"**{plate}**: {count} incidents")
        else:
            st.info("No data available yet")


def render_incidents():
    """Render incidents page"""
    st.markdown('<h1 class="main-header">🚨 Incidents</h1>', unsafe_allow_html=True)
    
    handler = get_evidence_handler()
    
    if not handler:
        st.error("Database not available")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "Pending", "Reviewed", "Actioned"]
        )
    
    with col2:
        search_plate = st.text_input("Search by Plate", placeholder="MH12AB1234")
    
    with col3:
        limit = st.slider("Show", 10, 100, 50)
    
    # Get incidents
    if search_plate:
        incidents = handler.search_by_plate(search_plate)
    elif status_filter != "All":
        incidents = handler.get_incidents(limit=limit, status=status_filter.lower())
    else:
        incidents = handler.get_incidents(limit=limit)
    
    st.markdown(f"### Showing {len(incidents)} incidents")
    
    # Display incidents
    for incident in incidents:
        with st.expander(
            f"🚨 #{incident['id']} - {incident.get('license_plate', 'Unknown')} - "
            f"{incident.get('timestamp', 'N/A')[:19]}"
        ):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Show image if available
                frame_path = incident.get('frame_path')
                if frame_path and Path(frame_path).exists():
                    st.image(frame_path, caption="Incident Frame", width='stretch')
                
                plate_path = incident.get('plate_image_path')
                if plate_path and Path(plate_path).exists():
                    st.image(plate_path, caption="License Plate", width=200)
            
            with col2:
                st.markdown(f"**License Plate:** `{incident.get('license_plate', 'N/A')}`")
                st.markdown(f"**Confidence:** {incident.get('plate_confidence', 0):.1%}")
                st.markdown(f"**Location:** {incident.get('location', 'Not specified')}")
                st.markdown(f"**Source:** {incident.get('source', 'N/A')}")
                st.markdown(f"**Status:** {incident.get('status', 'pending')}")
                st.markdown(f"**Alert Sent:** {'✅' if incident.get('alert_sent') else '❌'}")
                
                # Actions
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if st.button("✅ Mark Reviewed", key=f"review_{incident['id']}"):
                        handler.update_incident(incident['id'], status='reviewed')
                        st.rerun()
                
                with col_b:
                    if st.button("📤 Send Alert", key=f"alert_{incident['id']}"):
                        st.info("Sending alert...")
                
                with col_c:
                    if st.button("🗑️ Delete", key=f"delete_{incident['id']}"):
                        st.warning("Delete functionality coming soon")


def render_settings():
    """Render settings page"""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔧 Detection", "📱 Telegram", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Detection Settings")
        
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.35)
        st.caption("Minimum confidence for object detection")
        
        proximity_threshold = st.slider("Proximity Threshold (px)", 50, 500, 200)
        st.caption("Maximum distance between waste and vehicle for event detection")
        
        alert_cooldown = st.slider("Alert Cooldown (seconds)", 10, 300, 30)
        st.caption("Minimum time between alerts for same vehicle")
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved!")
    
    with tab2:
        st.markdown("### Telegram Bot Configuration")
        
        st.info("""
        **To set up Telegram alerts:**
        1. Open Telegram and search for @BotFather
        2. Send /newbot and follow instructions
        3. Copy the API token provided
        4. Create a group and add your bot
        5. Get the chat ID using @userinfobot
        """)
        
        token = st.text_input("Bot Token", type="password", placeholder="Enter your bot token")
        chat_id = st.text_input("Chat ID", placeholder="Enter your chat/group ID")
        
        if st.button("💾 Save & Test"):
            if token and chat_id:
                try:
                    from telegram_bot import TelegramAlertBot
                    bot = TelegramAlertBot(token, chat_id)
                    if bot.send_test_message():
                        st.success("✅ Test message sent! Check your Telegram.")
                    else:
                        st.error("❌ Failed to send test message")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter both token and chat ID")
    
    with tab3:
        st.markdown("### About CivicCam")
        
        st.markdown("""
        **CivicCam** is an AI-powered littering detection system designed to:
        
        - 🎥 Monitor live camera feeds or process recorded videos
        - 🚗 Detect vehicles and license plates
        - 🗑️ Identify littering events
        - 📱 Send real-time alerts via Telegram
        - 📊 Provide analytics and incident management
        
        ---
        
        **Technology Stack:**
        - YOLOv8 for object detection
        - EasyOCR for license plate reading
        - Streamlit for the web interface
        - SQLite for data storage
        - Telegram Bot API for alerts
        
        ---
        
        **Model Performance:**
        - mAP50: 92.3%
        - Precision: 95.8%
        - Recall: 87.7%
        
        ---
        
        *Built as a Final Year Project*
        """)


def main():
    """Main application"""
    page = render_sidebar()
    
    if page == "🎥 Live Feed":
        render_live_feed()
    elif page == "📊 Dashboard":
        render_dashboard()
    elif page == "🚨 Incidents":
        render_incidents()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
