"""
CivicCam Evidence Handler
Captures and stores evidence for littering incidents
"""

import cv2
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import shutil


class EvidenceHandler:
    """Handles evidence capture, storage, and retrieval for littering incidents"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize the evidence handler
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            from config import DATABASE_PATH, INCIDENTS_DIR
            self.db_path = DATABASE_PATH
            self.incidents_dir = INCIDENTS_DIR
        else:
            self.db_path = Path(db_path)
            self.incidents_dir = self.db_path.parent
        
        self.incidents_dir.mkdir(parents=True, exist_ok=True)
        (self.incidents_dir / "images").mkdir(exist_ok=True)
        (self.incidents_dir / "clips").mkdir(exist_ok=True)
        
        self._init_database()
        print(f"[EvidenceHandler] Database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                license_plate TEXT,
                plate_confidence REAL,
                location TEXT,
                source TEXT,
                frame_path TEXT,
                plate_image_path TEXT,
                waste_image_path TEXT,
                video_clip_path TEXT,
                detections TEXT,
                status TEXT DEFAULT 'pending',
                notes TEXT,
                alert_sent INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _sanitize_detections(self, detections: List[Dict]) -> List[Dict]:
        """Remove non-serializable objects (like numpy arrays) from detections"""
        sanitized = []
        for det in detections:
            clean_det = {}
            for k, v in det.items():
                if isinstance(v, (np.ndarray, bytes)):
                    continue  # Skip numpy arrays and bytes
                clean_det[k] = v
            sanitized.append(clean_det)
        return sanitized

    def save_incident(self, 
                     frame: np.ndarray,
                     license_plate: str,
                     plate_confidence: float,
                     detections: List[Dict],
                     source: str = "camera",
                     location: str = "",
                     plate_crop: np.ndarray = None,
                     waste_crop: np.ndarray = None) -> int:
        """
        Save a new littering incident
        
        Args:
            frame: Full frame image
            license_plate: Detected license plate text
            plate_confidence: OCR confidence
            detections: List of all detections in frame
            source: Source of the video/image
            location: Location description
            plate_crop: Cropped license plate image
            waste_crop: Cropped waste image
            
        Returns:
            Incident ID
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save full frame
        frame_filename = f"incident_{timestamp_str}.jpg"
        frame_path = self.incidents_dir / "images" / frame_filename
        cv2.imwrite(str(frame_path), frame)
        
        # Save plate crop if available
        plate_image_path = None
        if plate_crop is not None and plate_crop.size > 0:
            plate_filename = f"plate_{timestamp_str}.jpg"
            plate_image_path = self.incidents_dir / "images" / plate_filename
            cv2.imwrite(str(plate_image_path), plate_crop)
        
        # Save waste crop if available
        waste_image_path = None
        if waste_crop is not None and waste_crop.size > 0:
            waste_filename = f"waste_{timestamp_str}.jpg"
            waste_image_path = self.incidents_dir / "images" / waste_filename
            cv2.imwrite(str(waste_image_path), waste_crop)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO incidents 
            (timestamp, license_plate, plate_confidence, location, source, 
             frame_path, plate_image_path, waste_image_path, detections)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(),
            license_plate,
            plate_confidence,
            location,
            source,
            str(frame_path),
            str(plate_image_path) if plate_image_path else None,
            str(waste_image_path) if waste_image_path else None,
            json.dumps(self._sanitize_detections(detections))
        ))
        
        incident_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"[EvidenceHandler] Saved incident #{incident_id}: {license_plate}")
        return incident_id
    
    def get_incidents(self, limit: int = 50, status: str = None) -> List[Dict]:
        """
        Get recent incidents
        
        Args:
            limit: Maximum number of incidents to return
            status: Filter by status ('pending', 'reviewed', 'actioned')
            
        Returns:
            List of incident dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if status:
            cursor.execute('''
                SELECT * FROM incidents 
                WHERE status = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (status, limit))
        else:
            cursor.execute('''
                SELECT * FROM incidents 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_incident(self, incident_id: int) -> Optional[Dict]:
        """Get a specific incident by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM incidents WHERE id = ?', (incident_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def update_incident(self, incident_id: int, **kwargs) -> bool:
        """
        Update incident fields
        
        Args:
            incident_id: ID of incident to update
            **kwargs: Fields to update (status, notes, alert_sent, etc.)
            
        Returns:
            True if updated successfully
        """
        if not kwargs:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [incident_id]
        
        cursor.execute(f'''
            UPDATE incidents SET {set_clause} WHERE id = ?
        ''', values)
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    
    def mark_alert_sent(self, incident_id: int) -> bool:
        """Mark an incident as having alert sent"""
        return self.update_incident(incident_id, alert_sent=1)
    
    def get_stats(self) -> Dict:
        """Get incident statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total incidents
        cursor.execute('SELECT COUNT(*) FROM incidents')
        total = cursor.fetchone()[0]
        
        # By status
        cursor.execute('''
            SELECT status, COUNT(*) FROM incidents GROUP BY status
        ''')
        by_status = dict(cursor.fetchall())
        
        # Today's incidents
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute('''
            SELECT COUNT(*) FROM incidents WHERE DATE(timestamp) = ?
        ''', (today,))
        today_count = cursor.fetchone()[0]
        
        # Unique plates
        cursor.execute('''
            SELECT COUNT(DISTINCT license_plate) FROM incidents 
            WHERE license_plate IS NOT NULL AND license_plate != ''
        ''')
        unique_plates = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_incidents": total,
            "by_status": by_status,
            "today_count": today_count,
            "unique_plates": unique_plates
        }
    
    def search_by_plate(self, plate_query: str) -> List[Dict]:
        """Search incidents by license plate"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM incidents 
            WHERE license_plate LIKE ?
            ORDER BY timestamp DESC
        ''', (f'%{plate_query}%',))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]


if __name__ == "__main__":
    # Test the evidence handler
    handler = EvidenceHandler()
    
    # Create a test incident
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "TEST INCIDENT", (200, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    incident_id = handler.save_incident(
        frame=test_frame,
        license_plate="MH12AB1234",
        plate_confidence=0.95,
        detections=[{"class_name": "license_plate", "confidence": 0.95}],
        source="test",
        location="Test Location"
    )
    
    print(f"\nCreated test incident: #{incident_id}")
    print(f"\nStats: {handler.get_stats()}")
    print(f"\nRecent incidents: {len(handler.get_incidents())} found")
