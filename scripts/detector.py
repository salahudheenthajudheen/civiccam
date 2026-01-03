"""
CivicCam Detection Engine
Main detection module using YOLOv8 for object detection
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import time


class CivicCamDetector:
    """YOLOv8-based detector for CivicCam"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = None):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the trained model
            conf_threshold: Minimum confidence threshold (reads from config if None)
        """
        from config import MODEL_PATH, DETECTION_CONF
        
        if model_path is None:
            model_path = str(MODEL_PATH)
        
        if conf_threshold is None:
            conf_threshold = DETECTION_CONF
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        
        print(f"[CivicCamDetector] Loaded model from {model_path}")
        print(f"[CivicCamDetector] Confidence threshold: {conf_threshold}")
        print(f"[CivicCamDetector] Classes: {self.class_names}")
    
    def detect(self, frame: np.ndarray, conf: float = None) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: BGR image as numpy array
            conf: Confidence threshold (uses default if None)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - class_id: int
            - class_name: str
            - confidence: float
        """
        conf = conf or self.conf_threshold
        
        results = self.model(frame, conf=conf, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": confidence
            })
        
        return detections
    
    def detect_and_draw(self, frame: np.ndarray, conf: float = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects and draw bounding boxes on frame
        
        Args:
            frame: BGR image as numpy array
            conf: Confidence threshold
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        from config import CLASS_COLORS
        
        detections = self.detect(frame, conf)
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            
            # Get color for this class
            color = CLASS_COLORS.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated, detections
    
    def process_video(self, video_path: str, output_path: str = None, 
                     callback=None, max_frames: int = None) -> List[Dict]:
        """
        Process a video file and detect objects in each frame
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            callback: Function to call for each frame (frame, detections)
            max_frames: Maximum frames to process (for testing)
            
        Returns:
            List of all detections with frame numbers
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[Video] {video_path}: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_num >= max_frames:
                break
            
            # Detect objects
            annotated, detections = self.detect_and_draw(frame)
            
            # Add frame number to detections
            for det in detections:
                det["frame"] = frame_num
                det["timestamp"] = frame_num / fps
            
            all_detections.extend(detections)
            
            # Call callback if provided
            if callback:
                callback(annotated, detections, frame_num)
            
            # Write to output
            if writer:
                writer.write(annotated)
            
            frame_num += 1
            
            if frame_num % 100 == 0:
                print(f"[Video] Processed {frame_num}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
        
        print(f"[Video] Complete! Processed {frame_num} frames, found {len(all_detections)} detections")
        return all_detections
    
    def get_license_plates(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections to only license plates"""
        return [d for d in detections if d["class_name"] == "license_plate"]
    
    def get_waste(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections to only waste"""
        return [d for d in detections if d["class_name"] == "waste"]


if __name__ == "__main__":
    # Test the detector
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    detector = CivicCamDetector()
    
    # Test with a sample image if available
    test_images = list(Path("datasets/combined/valid/images").glob("*.jpg"))[:3]
    
    for img_path in test_images:
        print(f"\nTesting: {img_path.name}")
        frame = cv2.imread(str(img_path))
        detections = detector.detect(frame)
        
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")
