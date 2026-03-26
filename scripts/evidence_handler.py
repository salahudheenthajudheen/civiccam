"""
CivicCam Evidence Handler
Captures and stores evidence for littering incidents, with Supabase integration.
"""

import cv2
import numpy as np
import sqlite3
import os
import io
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
            db_path: Path to SQLite database (fallback)
        """
        from config import SUPABASE_URL, SUPABASE_KEY, DATABASE_PATH, INCIDENTS_DIR
        
        self.supabase_url = SUPABASE_URL
        self.supabase_key = SUPABASE_KEY
        
        self.use_supabase = bool(self.supabase_url and self.supabase_key)
        
        if self.use_supabase:
            try:
                from supabase import create_client, Client
                self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
                print("[EvidenceHandler] Connected to Supabase backend.")
            except ImportError:
                print("[EvidenceHandler] 'supabase' package not installed. Falling back to SQLite.")
                self.use_supabase = False
            except Exception as e:
                print(f"[EvidenceHandler] Failed to initialize Supabase: {e}. Falling back to SQLite.")
                self.use_supabase = False
        
        if not self.use_supabase:
            print("[EvidenceHandler] Using local SQLite fallback.")
            if db_path is None:
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
        """Initialize SQLite database (fallback)"""
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
            )''')
        
        conn.commit()
        conn.close()
    
    def _sanitize_detections(self, detections: List[Dict]) -> List[Dict]:
        """Remove non-serializable objects (like numpy arrays) from detections"""
        sanitized = []
        for det in detections:
            clean_det = {}
            for k, v in det.items():
                if isinstance(v, (np.ndarray, bytes)):
                    continue
                clean_det[k] = v
            sanitized.append(clean_det)
        return sanitized

    def _upload_image_to_supabase(self, image: np.ndarray, filename: str) -> Optional[str]:
        """Helper to encode and upload an image to Supabase Storage"""
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            return None
        
        try:
            res = self.supabase.storage.from_("incident-images").upload(
                file=buffer.tobytes(),
                path=filename,
                file_options={"content-type": "image/jpeg"}
            )
            return self.supabase.storage.from_("incident-images").get_public_url(filename)
        except Exception as e:
            print(f"[EvidenceHandler] Supabase upload failed for {filename}: {e}")
            return None

    def save_incident(self, 
                     frame: np.ndarray,
                     license_plate: str,
                     plate_confidence: float,
                     detections: List[Dict],
                     source: str = "camera",
                     location: str = "",
                     plate_crop: np.ndarray = None,
                     waste_crop: np.ndarray = None) -> int:
        
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        clean_detections = self._sanitize_detections(detections)
        
        frame_filename = f"incident_{timestamp_str}.jpg"
        plate_filename = f"plate_{timestamp_str}.jpg" if (plate_crop is not None and plate_crop.size > 0) else None
        waste_filename = f"waste_{timestamp_str}.jpg" if (waste_crop is not None and waste_crop.size > 0) else None
        
        if self.use_supabase:
            # Upload to Supabase Storage
            frame_url = self._upload_image_to_supabase(frame, frame_filename)
            plate_url = self._upload_image_to_supabase(plate_crop, plate_filename) if plate_filename else None
            waste_url = self._upload_image_to_supabase(waste_crop, waste_filename) if waste_filename else None
            
            # Insert into Supabase logic
            try:
                data = {
                    "timestamp": timestamp.isoformat(),
                    "license_plate": license_plate,
                    "plate_confidence": plate_confidence,
                    "location": location,
                    "source": source,
                    "frame_url": frame_url,
                    "plate_image_url": plate_url,
                    "waste_image_url": waste_url,
                    "detections": clean_detections,
                    "status": "pending",
                    "alert_sent": False
                }
                result = self.supabase.table("incidents").insert(data).execute()
                incident_id = result.data[0]["id"]
                print(f"[EvidenceHandler] Saved incident #{incident_id} to Supabase: {license_plate}")
                return incident_id
            except Exception as e:
                print(f"[EvidenceHandler] DB insert failed: {e}")
                return -1

        else:
            # Fallback local SQLite save logic
            frame_path = self.incidents_dir / "images" / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            plate_image_path = None
            if plate_filename:
                plate_image_path = self.incidents_dir / "images" / plate_filename
                cv2.imwrite(str(plate_image_path), plate_crop)
            
            waste_image_path = None
            if waste_filename:
                waste_image_path = self.incidents_dir / "images" / waste_filename
                cv2.imwrite(str(waste_image_path), waste_crop)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO incidents 
                (timestamp, license_plate, plate_confidence, location, source, 
                 frame_path, plate_image_path, waste_image_path, detections)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp.isoformat(), license_plate, plate_confidence, location, source, 
                str(frame_path), str(plate_image_path) if plate_image_path else None, 
                str(waste_image_path) if waste_image_path else None, json.dumps(clean_detections)
            ))
            
            incident_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"[EvidenceHandler] Saved incident #{incident_id} locally: {license_plate}")
            return incident_id
    
    def get_incidents(self, limit: int = 50, status: str = None) -> List[Dict]:
        if self.use_supabase:
            try:
                query = self.supabase.table("incidents").select("*").order("timestamp", desc=True).limit(limit)
                if status:
                    query = query.eq("status", status)
                res = query.execute()
                return res.data
            except Exception as e:
                print(f"[EvidenceHandler] Error fetching incidents: {e}")
                return []
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if status:
                cursor.execute('SELECT * FROM incidents WHERE status = ? ORDER BY timestamp DESC LIMIT ?', (status, limit))
            else:
                cursor.execute('SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
    
    def get_incident(self, incident_id: int) -> Optional[Dict]:
        if self.use_supabase:
            try:
                res = self.supabase.table("incidents").select("*").eq("id", incident_id).execute()
                return res.data[0] if res.data else None
            except Exception as e:
                print(f"[EvidenceHandler] Error fetching incident: {e}")
                return None
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM incidents WHERE id = ?', (incident_id,))
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
    
    def update_incident(self, incident_id: int, **kwargs) -> bool:
        if not kwargs: return False
        
        if self.use_supabase:
            try:
                res = self.supabase.table("incidents").update(kwargs).eq("id", incident_id).execute()
                return len(res.data) > 0
            except Exception as e:
                print(f"[EvidenceHandler] Error updating incident: {e}")
                return False
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [incident_id]
            cursor.execute(f'UPDATE incidents SET {set_clause} WHERE id = ?', values)
            conn.commit()
            success = cursor.rowcount > 0
            conn.close()
            return success
    
    def mark_alert_sent(self, incident_id: int) -> bool:
        if self.use_supabase:
            return self.update_incident(incident_id, alert_sent=True)
        else:
            return self.update_incident(incident_id, alert_sent=1)
    
    def get_stats(self) -> Dict:
        if self.use_supabase:
            try:
                res = self.supabase.table("incidents").select("status, timestamp, license_plate").execute()
                data = res.data
                total = len(data)
                
                by_status = {}
                today = datetime.now().strftime("%Y-%m-%d")
                today_count = 0
                plates = set()
                
                for item in data:
                    s = item.get("status", "pending")
                    by_status[s] = by_status.get(s, 0) + 1
                    
                    t = item.get("timestamp", "")
                    if t.startswith(today):
                        today_count += 1
                        
                    p = item.get("license_plate")
                    if p:
                        plates.add(p)
                
                return {
                    "total_incidents": total,
                    "by_status": by_status,
                    "today_count": today_count,
                    "unique_plates": len(plates)
                }
            except Exception as e:
                print(f"[EvidenceHandler] Error compiling stats: {e}")
                return {"total_incidents": 0, "by_status": {}, "today_count": 0, "unique_plates": 0}
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM incidents')
            total = cursor.fetchone()[0]
            
            cursor.execute('SELECT status, COUNT(*) FROM incidents GROUP BY status')
            by_status = dict(cursor.fetchall())
            
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute('SELECT COUNT(*) FROM incidents WHERE DATE(timestamp) = ?', (today,))
            today_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT license_plate) FROM incidents WHERE license_plate IS NOT NULL AND license_plate != ''")
            unique_plates = cursor.fetchone()[0]
            conn.close()
            
            return {
                "total_incidents": total,
                "by_status": by_status,
                "today_count": today_count,
                "unique_plates": unique_plates
            }
    
    def search_by_plate(self, plate_query: str) -> List[Dict]:
        if self.use_supabase:
            try:
                res = self.supabase.table("incidents").select("*").ilike("license_plate", f"%{plate_query}%").order("timestamp", desc=True).execute()
                return res.data
            except Exception as e:
                print(f"[EvidenceHandler] Error searching plate: {e}")
                return []
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents WHERE license_plate LIKE ? ORDER BY timestamp DESC", (f'%{plate_query}%',))
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]


if __name__ == "__main__":
    handler = EvidenceHandler()
    
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
