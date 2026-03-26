"""
CivicCam Littering Event Detector
Detects littering events by analyzing object relationships across frames
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import time


class LitteringEventDetector:
    """Detects littering events from object detections"""
    
    def __init__(self, 
                 proximity_threshold: int = 200,
                 time_window: float = 5.0):
        """
        Initialize the littering event detector
        
        Args:
            proximity_threshold: Max pixels between waste and vehicle/person
            time_window: Seconds to track for event detection
        """
        self.proximity_threshold = proximity_threshold
        self.time_window = time_window
        
        # Tracking state
        self.tracked_objects = defaultdict(list)  # {class_name: [(bbox, timestamp), ...]}
        self.detected_events = []
        self.last_cleanup = time.time()
        
        # Track previous waste locations to detect NEW waste (dumping)
        self.previous_waste_bboxes = []  # [(bbox, timestamp), ...]
        self.waste_iou_threshold = 0.3  # If IOU > this, it's the same waste
        self.new_waste_cooldown = 5.0  # Seconds before same location can trigger again
        
        print(f"[EventDetector] Proximity: {proximity_threshold}px, Window: {time_window}s")
    
    def _calculate_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate distance between two bounding box centers"""
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def _cleanup_old_tracks(self, current_time: float):
        """Remove old tracking data"""
        if current_time - self.last_cleanup < 1.0:  # Cleanup every second
            return
        
        cutoff = current_time - self.time_window
        for class_name in list(self.tracked_objects.keys()):
            self.tracked_objects[class_name] = [
                (bbox, ts) for bbox, ts in self.tracked_objects[class_name]
                if ts > cutoff
            ]
            if not self.tracked_objects[class_name]:
                del self.tracked_objects[class_name]
        
        # Also cleanup old waste tracking
        waste_cutoff = current_time - self.new_waste_cooldown
        self.previous_waste_bboxes = [
            (bbox, ts) for bbox, ts in self.previous_waste_bboxes
            if ts > waste_cutoff
        ]
        
        self.last_cleanup = current_time
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_new_waste(self, waste_bbox: List[int], current_time: float) -> bool:
        """Check if this waste is NEW (just appeared) or was already there"""
        for prev_bbox, prev_time in self.previous_waste_bboxes:
            iou = self._calculate_iou(waste_bbox, prev_bbox)
            if iou > self.waste_iou_threshold:
                # Same waste, not new
                return False
        
        # This is NEW waste - add to tracking
        self.previous_waste_bboxes.append((waste_bbox, current_time))
        return True

    
    def process_detections(self, 
                          detections: List[Dict],
                          timestamp: float = None) -> List[Dict]:
        """
        Process detections and identify potential littering events
        
        Args:
            detections: List of detection dictionaries
            timestamp: Current timestamp (uses system time if None)
            
        Returns:
            List of littering event dictionaries
        """
        current_time = timestamp or time.time()
        self._cleanup_old_tracks(current_time)
        
        # Separate detections by class
        license_plates = [d for d in detections if d["class_name"] == "license_plate"]
        waste_objects = [d for d in detections if d["class_name"] == "waste" or d["class_name"] == "Waste"]
        objects = [d for d in detections if d["class_name"] == "object"]
        people = [d for d in detections if d["class_name"] == "public"]
        faces = [d for d in detections if d["class_name"] == "face"]
        
        # Track all detections
        for det in detections:
            self.tracked_objects[det["class_name"]].append(
                (det["bbox"], current_time)
            )
        
        # Find littering events - only for NEW waste (dumping action)
        events = []
        
        for waste in waste_objects:
            waste_bbox = waste["bbox"]
            
            # Only process if this is NEW waste (just appeared = dumping)
            is_new = self._is_new_waste(waste_bbox, current_time)
            
            if not is_new:
                # This waste was already there, not a dumping event
                continue
            
            print(f"[EventDetector] NEW WASTE DETECTED at {waste_bbox}")
            
            event_triggered = False
            
            # Check proximity to license plates (vehicles)
            for plate in license_plates:
                plate_bbox = plate["bbox"]
                distance = self._calculate_distance(waste_bbox, plate_bbox)
                
                if distance < self.proximity_threshold:
                    event = {
                        "type": "vehicle_dumping",
                        "timestamp": current_time,
                        "waste_bbox": waste_bbox,
                        "plate_bbox": plate_bbox,
                        "distance": distance,
                        "confidence": min(waste["confidence"], plate["confidence"]),
                        "is_new_waste": True,
                        "suspect_type": "vehicle",
                        "detections": {
                            "waste": waste,
                            "license_plate": plate
                        }
                    }
                    events.append(event)
                    event_triggered = True
                    
                    print(f"[EventDetector] 🚨 VEHICLE DUMPING DETECTED! "
                          f"New waste near plate - Distance: {distance:.0f}px, "
                          f"Confidence: {event['confidence']:.2f}")
            
            # Check proximity to faces (pedestrians) - only if no vehicle event
            if not event_triggered:
                for face in faces:
                    face_bbox = face["bbox"]
                    distance = self._calculate_distance(waste_bbox, face_bbox)
                    
                    if distance < self.proximity_threshold:
                        event = {
                            "type": "pedestrian_dumping",
                            "timestamp": current_time,
                            "waste_bbox": waste_bbox,
                            "face_bbox": face_bbox,
                            "distance": distance,
                            "confidence": min(waste["confidence"], face["confidence"]),
                            "is_new_waste": True,
                            "suspect_type": "pedestrian_face",
                            "detections": {
                                "waste": waste,
                                "face": face
                            }
                        }
                        events.append(event)
                        event_triggered = True
                        
                        print(f"[EventDetector] 🚨 PEDESTRIAN DUMPING DETECTED (face)! "
                              f"New waste near face - Distance: {distance:.0f}px, "
                              f"Confidence: {event['confidence']:.2f}")
                        break  # One face match is enough
            
            # Check proximity to people (public class) - fallback if no face detected
            if not event_triggered:
                for person in people:
                    person_bbox = person["bbox"]
                    distance = self._calculate_distance(waste_bbox, person_bbox)
                    
                    if distance < self.proximity_threshold:
                        event = {
                            "type": "pedestrian_dumping",
                            "timestamp": current_time,
                            "waste_bbox": waste_bbox,
                            "person_bbox": person_bbox,
                            "distance": distance,
                            "confidence": min(waste["confidence"], person["confidence"]),
                            "is_new_waste": True,
                            "suspect_type": "pedestrian",
                            "detections": {
                                "waste": waste,
                                "public": person
                            }
                        }
                        events.append(event)
                        event_triggered = True
                        
                        print(f"[EventDetector] 🚨 PEDESTRIAN DUMPING DETECTED! "
                              f"New waste near person - Distance: {distance:.0f}px, "
                              f"Confidence: {event['confidence']:.2f}")
                        break  # One person match is enough
        
        # Also check for objects being thrown (object + waste appearing together)
        for obj in objects:
            obj_bbox = obj["bbox"]
            
            for waste in waste_objects:
                waste_bbox = waste["bbox"]
                distance = self._calculate_distance(obj_bbox, waste_bbox)
                
                if distance < self.proximity_threshold * 0.5:  # Closer threshold for throwing
                    # Check if there's a nearby license plate
                    nearby_plate = None
                    for plate in license_plates:
                        if self._calculate_distance(obj_bbox, plate["bbox"]) < self.proximity_threshold * 1.5:
                            nearby_plate = plate
                            break
                    
                    if nearby_plate:
                        event = {
                            "type": "throwing",
                            "timestamp": current_time,
                            "object_bbox": obj_bbox,
                            "waste_bbox": waste_bbox,
                            "plate_bbox": nearby_plate["bbox"],
                            "distance": distance,
                            "confidence": min(obj["confidence"], waste["confidence"]),
                            "detections": {
                                "object": obj,
                                "waste": waste,
                                "license_plate": nearby_plate
                            }
                        }
                        events.append(event)
                        
                        print(f"[EventDetector] THROWING DETECTED! "
                              f"Distance: {distance:.0f}px")
        
        self.detected_events.extend(events)
        return events
    
    def get_recent_events(self, seconds: float = 60.0) -> List[Dict]:
        """Get events from the last N seconds"""
        current_time = time.time()
        cutoff = current_time - seconds
        
        return [e for e in self.detected_events if e["timestamp"] > cutoff]
    
    def clear_events(self):
        """Clear all detected events"""
        self.detected_events.clear()
        self.tracked_objects.clear()


class CivicCamPipeline:
    """Complete pipeline for CivicCam littering detection"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the complete pipeline
        
        Args:
            model_path: Path to YOLO model
        """
        from detector import CivicCamDetector
        from ocr_engine import LicensePlateOCR
        from evidence_handler import EvidenceHandler
        from telegram_bot import TelegramAlertBot
        
        self.detector = CivicCamDetector(model_path)
        self.ocr = LicensePlateOCR()
        self.evidence = EvidenceHandler()
        self.telegram = TelegramAlertBot()
        self.event_detector = LitteringEventDetector()
        
        self.alert_cooldown = {}  # {plate: last_alert_time}
        self.cooldown_seconds = 30
        
        print("[CivicCamPipeline] All components initialized")
    
    def process_frame(self, 
                     frame: np.ndarray,
                     source: str = "camera",
                     location: str = "") -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: BGR image
            source: Source identifier
            location: Location description
            
        Returns:
            Tuple of (annotated_frame, detections, events)
        """
        # Detect objects
        annotated, detections = self.detector.detect_and_draw(frame)
        
        # Process license plates with OCR
        for det in detections:
            if det["class_name"] == "license_plate":
                text, conf, plate_img = self.ocr.extract_plate_from_frame(
                    frame, det["bbox"]
                )
                det["plate_text"] = text
                det["ocr_confidence"] = conf
                det["plate_image"] = plate_img
        
        # Detect littering events
        events = self.event_detector.process_detections(detections)
        
        # Handle events
        for event in events:
            self._handle_event(event, frame, source, location)
        
        return annotated, detections, events
    
    def _handle_event(self, event: Dict, frame: np.ndarray, 
                     source: str, location: str):
        """Handle a detected littering event"""
        suspect_type = event.get("suspect_type", "unknown")
        
        # Get identifier and confidence based on event type
        if suspect_type == "vehicle":
            plate_det = event["detections"].get("license_plate", {})
            identifier = plate_det.get("plate_text", "UNKNOWN")
            confidence = plate_det.get("ocr_confidence", 0)
            suspect_crop = plate_det.get("plate_image")
        elif suspect_type == "pedestrian_face":
            face_det = event["detections"].get("face", {})
            identifier = f"PEDESTRIAN_FACE_{int(time.time())}"
            confidence = face_det.get("confidence", 0)
            # Crop the face
            if face_det and "bbox" in face_det:
                x1, y1, x2, y2 = [int(c) for c in face_det["bbox"]]
                suspect_crop = frame[y1:y2, x1:x2]
            else:
                suspect_crop = None
        else:  # pedestrian (body only)
            person_det = event["detections"].get("public", {})
            identifier = f"PEDESTRIAN_{int(time.time())}"
            confidence = person_det.get("confidence", 0)
            # Crop the person
            if person_det and "bbox" in person_det:
                x1, y1, x2, y2 = [int(c) for c in person_det["bbox"]]
                suspect_crop = frame[y1:y2, x1:x2]
            else:
                suspect_crop = None
        
        # Check cooldown
        current_time = time.time()
        if identifier in self.alert_cooldown:
            if current_time - self.alert_cooldown[identifier] < self.cooldown_seconds:
                return  # Skip, already alerted recently
        
        self.alert_cooldown[identifier] = current_time
        
        # Get waste crop
        waste_det = event["detections"].get("waste", {})
        waste_img = None
        if waste_det and "bbox" in waste_det:
            x1, y1, x2, y2 = [int(c) for c in waste_det["bbox"]]
            waste_img = frame[y1:y2, x1:x2]
        
        # Save evidence
        incident_id = self.evidence.save_incident(
            frame=frame,
            license_plate=identifier,
            plate_confidence=confidence,
            detections=[event["detections"]],
            source=source,
            location=location,
            plate_crop=suspect_crop,
            waste_crop=waste_img
        )
        
        # Send Telegram alert
        if self.telegram.is_configured():
            incident = self.evidence.get_incident(incident_id)
            self.telegram.send_alert(
                license_plate=identifier,
                confidence=confidence,
                location=location,
                image_path=incident.get("frame_path"),
                incident_id=incident_id
            )
            self.evidence.mark_alert_sent(incident_id)
        
        print(f"[Pipeline] Event handled: Suspect={identifier}, Type={suspect_type}, ID={incident_id}")


if __name__ == "__main__":
    # Test the event detector
    detector = LitteringEventDetector()
    
    # Simulate detections
    test_detections = [
        {"class_name": "license_plate", "bbox": [100, 100, 200, 150], "confidence": 0.95},
        {"class_name": "waste", "bbox": [150, 180, 200, 220], "confidence": 0.85},
    ]
    
    events = detector.process_detections(test_detections)
    print(f"\nDetected {len(events)} events")
    
    for event in events:
        print(f"  - Type: {event['type']}, Distance: {event['distance']:.0f}px")
