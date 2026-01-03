"""
CivicCam Face Detection Module
Uses YOLOv8 for face detection to identify litterers
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import urllib.request
import os


class FaceDetector:
    """Face detector using YOLOv8-face or OpenCV cascade"""
    
    def __init__(self, conf_threshold: float = 0.5, use_gpu: bool = True):
        """
        Initialize the face detector
        
        Args:
            conf_threshold: Minimum confidence for face detection
            use_gpu: Whether to use GPU acceleration
        """
        self.conf_threshold = conf_threshold
        self.use_gpu = use_gpu
        self.model = None
        self.cascade = None
        
        # Try YOLOv8 first, fall back to OpenCV cascade
        if self._init_yolo():
            self.method = "yolov8"
            print("[FaceDetector] Using YOLOv8 face detection")
        else:
            self._init_cascade()
            self.method = "cascade"
            print("[FaceDetector] Using OpenCV Haar Cascade (fallback)")
    
    def _init_yolo(self) -> bool:
        """Initialize YOLOv8 face detector"""
        try:
            from ultralytics import YOLO
            
            # Use yolov8n for face detection (general YOLO works for faces too)
            # We'll use the person class from COCO and crop face region
            self.model = YOLO("yolov8n.pt")
            return True
        except Exception as e:
            print(f"[FaceDetector] YOLOv8 init failed: {e}")
            return False
    
    def _init_cascade(self):
        """Initialize OpenCV Haar Cascade for face detection"""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            print("[FaceDetector] Warning: Could not load Haar cascade")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a frame
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of face detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - face_image: cropped face numpy array
        """
        if self.method == "yolov8":
            return self._detect_yolo(frame)
        else:
            return self._detect_cascade(frame)
    
    def _detect_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using YOLOv8 (detecting persons and estimating face region)"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        faces = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            # Only process "person" detections
            if class_name != "person":
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            
            # Estimate face region (top 25% of person bounding box)
            person_height = y2 - y1
            face_height = person_height * 0.25
            face_y2 = y1 + face_height
            
            # Make face box slightly narrower (center 60%)
            person_width = x2 - x1
            face_margin = person_width * 0.2
            face_x1 = x1 + face_margin
            face_x2 = x2 - face_margin
            
            face_bbox = [int(face_x1), int(y1), int(face_x2), int(face_y2)]
            
            # Crop face
            fx1, fy1, fx2, fy2 = face_bbox
            fx1 = max(0, fx1)
            fy1 = max(0, fy1)
            fx2 = min(frame.shape[1], fx2)
            fy2 = min(frame.shape[0], fy2)
            
            face_img = frame[fy1:fy2, fx1:fx2].copy() if fy2 > fy1 and fx2 > fx1 else None
            
            faces.append({
                "bbox": face_bbox,
                "class_name": "face",
                "class_id": -1,  # Special ID for face
                "confidence": confidence,
                "face_image": face_img,
            })
        
        return faces
    
    def _detect_cascade(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade"""
        if self.cascade is None or self.cascade.empty():
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in detections:
            face_bbox = [int(x), int(y), int(x + w), int(y + h)]
            
            # Crop face
            face_img = frame[y:y+h, x:x+w].copy()
            
            faces.append({
                "bbox": face_bbox,
                "class_name": "face",
                "class_id": -1,
                "confidence": 0.8,  # Cascade doesn't provide confidence
                "face_image": face_img,
            })
        
        return faces


# Test the face detector
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    detector = FaceDetector()
    
    # Test on sample images
    test_images = list(Path("datasets/combined_v2/valid/images").glob("*.jpg"))[:5]
    
    for img_path in test_images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        faces = detector.detect_faces(frame)
        print(f"\n{img_path.name}: {len(faces)} faces detected")
        
        for i, face in enumerate(faces):
            print(f"  Face {i+1}: conf={face['confidence']:.2f}, bbox={face['bbox']}")
