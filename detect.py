"""
CivicCam Detection CLI
Unified detection script for images, videos, and webcam
Usage:
    python detect.py --source image.jpg
    python detect.py --source video.mp4
    python detect.py --source 0  (webcam)
    python detect.py --source ./images/  (directory)
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from config import MODEL_PATH, DETECTION_CONF, CLASS_COLORS, INCIDENTS_DIR, FACE_CONF_THRESHOLD


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CivicCam Object Detection")
    
    parser.add_argument("--source", type=str, default="0",
                       help="Input source: image path, video path, webcam index, or directory")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to YOLO model (default: models/civiccam_best.pt)")
    parser.add_argument("--conf", type=float, default=DETECTION_CONF,
                       help=f"Confidence threshold (default: {DETECTION_CONF})")
    parser.add_argument("--save", action="store_true",
                       help="Save detection results")
    parser.add_argument("--save-txt", action="store_true",
                       help="Save results as YOLO format labels")
    parser.add_argument("--show", action="store_true",
                       help="Display detection window")
    parser.add_argument("--ocr", action="store_true",
                       help="Enable license plate OCR")
    parser.add_argument("--events", action="store_true",
                       help="Enable littering event detection")
    parser.add_argument("--face", action="store_true",
                       help="Enable face detection")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: runs/detect/predict)")
    
    return parser.parse_args()


class CivicCamDetect:
    """Unified detection class for CivicCam"""
    
    def __init__(self, model_path=None, conf=0.35, enable_ocr=False, enable_events=False, enable_face=False):
        from ultralytics import YOLO
        
        # Load model
        model_path = model_path or str(MODEL_PATH)
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names
        
        print(f"[CivicCam] Model loaded: {model_path}")
        print(f"[CivicCam] Classes: {list(self.class_names.values())}")
        
        # OCR engine
        self.ocr = None
        if enable_ocr:
            try:
                from scripts.ocr_engine import LicensePlateOCR
                self.ocr = LicensePlateOCR()
                print("[CivicCam] OCR engine enabled")
            except Exception as e:
                print(f"[CivicCam] OCR not available: {e}")
        
        # Event detector
        self.event_detector = None
        if enable_events:
            try:
                from scripts.event_detector import LitteringEventDetector
                self.event_detector = LitteringEventDetector()
                print("[CivicCam] Event detection enabled")
            except Exception as e:
                print(f"[CivicCam] Event detection not available: {e}")
        
        # Face detector
        self.face_detector = None
        if enable_face:
            try:
                from scripts.face_detector import FaceDetector
                self.face_detector = FaceDetector(conf_threshold=FACE_CONF_THRESHOLD)
                print("[CivicCam] Face detection enabled")
            except Exception as e:
                print(f"[CivicCam] Face detection not available: {e}")
        
        # Stats
        self.stats = {
            "frames": 0,
            "detections": 0,
            "plates": 0,
            "waste": 0,
            "faces": 0,
            "events": 0,
        }
    
    def detect(self, frame):
        """Run detection on a single frame"""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            det = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": confidence,
            }
            
            # OCR for license plates
            if det["class_name"] == "license_plate" and self.ocr:
                text, ocr_conf, _ = self.ocr.extract_plate_from_frame(frame, det["bbox"])
                det["plate_text"] = text
                det["ocr_confidence"] = ocr_conf
            
            detections.append(det)
        
        # Detect faces if enabled
        if self.face_detector:
            faces = self.face_detector.detect_faces(frame)
            detections.extend(faces)
        
        # Update stats
        self.stats["frames"] += 1
        self.stats["detections"] += len(detections)
        self.stats["plates"] += sum(1 for d in detections if d["class_name"] == "license_plate")
        self.stats["waste"] += sum(1 for d in detections if d["class_name"] == "waste")
        self.stats["faces"] += sum(1 for d in detections if d["class_name"] == "face")
        
        # Check for littering events
        events = []
        if self.event_detector and detections:
            events = self.event_detector.process_detections(detections)
            self.stats["events"] += len(events)
        
        return detections, events
    
    def draw_detections(self, frame, detections, events=None):
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            
            # Get color
            color = CLASS_COLORS.get(class_name, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label text
            label = f"{class_name}: {confidence:.2f}"
            if "plate_text" in det and det["plate_text"]:
                label = f"{det['plate_text']} ({det['ocr_confidence']:.0%})"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw event alerts
        if events:
            for i, event in enumerate(events):
                alert_text = f"LITTERING DETECTED!"
                cv2.putText(annotated, alert_text, (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return annotated
    
    def process_image(self, image_path, save_dir=None, show=False, save_txt=False):
        """Process a single image"""
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[Error] Cannot read image: {image_path}")
            return
        
        detections, events = self.detect(frame)
        annotated = self.draw_detections(frame, detections, events)
        
        # Print results
        print(f"\n[Image] {Path(image_path).name}: {len(detections)} detections")
        for det in detections:
            plate_info = f" ({det.get('plate_text', '')})" if det.get('plate_text') else ""
            print(f"   - {det['class_name']}: {det['confidence']:.0%}{plate_info}")
        
        # Save results
        if save_dir:
            save_path = save_dir / Path(image_path).name
            cv2.imwrite(str(save_path), annotated)
            print(f"   Saved: {save_path}")
            
            if save_txt and detections:
                txt_path = save_dir / (Path(image_path).stem + ".txt")
                self._save_labels(txt_path, detections, frame.shape)
        
        # Display
        if show:
            cv2.imshow("CivicCam Detection", annotated)
            cv2.waitKey(0)
        
        return detections, events
    
    def process_video(self, video_path, save_dir=None, show=False):
        """Process a video file"""
        cap = cv2.VideoCapture(str(video_path) if str(video_path) != "0" else 0)
        
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        is_webcam = str(video_path) == "0"
        source_name = "Webcam" if is_webcam else Path(video_path).name
        print(f"\n[Video] {source_name}: {width}x{height} @ {fps}fps")
        
        # Setup output writer
        writer = None
        if save_dir and not is_webcam:
            output_path = save_dir / Path(video_path).name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect
                detections, events = self.detect(frame)
                annotated = self.draw_detections(frame, detections, events)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Draw FPS counter
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(annotated, fps_text, (width - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Progress for video files
                if not is_webcam and total_frames > 0:
                    progress = frame_count / total_frames * 100
                    if frame_count % 100 == 0:
                        print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # Write output
                if writer:
                    writer.write(annotated)
                
                # Display
                if show:
                    cv2.imshow("CivicCam Detection", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # Q or ESC
                        print("\n[Info] Detection stopped by user")
                        break
                
                frame_count += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n[Complete] Processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")
        print(f"   Detections: {self.stats['detections']}")
        print(f"   License plates: {self.stats['plates']}")
        print(f"   Waste objects: {self.stats['waste']}")
        print(f"   Faces: {self.stats['faces']}")
        print(f"   Littering events: {self.stats['events']}")
        
        if save_dir and not is_webcam:
            print(f"   Saved to: {save_dir}")
    
    def process_directory(self, dir_path, save_dir=None, show=False, save_txt=False):
        """Process all images in a directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in Path(dir_path).iterdir() 
                  if f.suffix.lower() in image_extensions]
        
        print(f"\n[Directory] Found {len(images)} images in {dir_path}")
        
        for i, img_path in enumerate(images):
            print(f"\n[{i+1}/{len(images)}] Processing: {img_path.name}")
            self.process_image(img_path, save_dir, show=False, save_txt=save_txt)
        
        # Summary
        print(f"\n[Complete] Processed {len(images)} images")
        print(f"   Total detections: {self.stats['detections']}")
        print(f"   License plates: {self.stats['plates']}")
        print(f"   Waste objects: {self.stats['waste']}")
    
    def _save_labels(self, txt_path, detections, frame_shape):
        """Save detections in YOLO format"""
        h, w = frame_shape[:2]
        
        with open(txt_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CivicCam Detection System")
    print("=" * 60)
    
    # Create output directory
    if args.save:
        output_dir = Path(args.output) if args.output else Path("runs/detect/predict")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = None
    
    # Initialize detector
    detector = CivicCamDetect(
        model_path=args.model,
        conf=args.conf,
        enable_ocr=args.ocr,
        enable_events=args.events,
        enable_face=args.face,
    )
    
    # Determine source type
    source = args.source
    source_path = Path(source) if source not in ["0", "1", "2"] else None
    
    if source.isdigit():
        # Webcam
        print(f"\n[Mode] Webcam {source}")
        detector.process_video(source, save_dir=output_dir, show=True)
        
    elif source_path and source_path.is_dir():
        # Directory
        print(f"\n[Mode] Directory batch processing")
        detector.process_directory(source_path, save_dir=output_dir, 
                                  show=args.show, save_txt=args.save_txt)
        
    elif source_path and source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        # Video file
        print(f"\n[Mode] Video file")
        detector.process_video(source_path, save_dir=output_dir, show=args.show)
        
    elif source_path and source_path.is_file():
        # Single image
        print(f"\n[Mode] Single image")
        detector.process_image(source_path, save_dir=output_dir, 
                              show=args.show, save_txt=args.save_txt)
    else:
        print(f"[Error] Invalid source: {source}")
        print("  Valid sources: image path, video path, webcam index (0,1,2), or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
