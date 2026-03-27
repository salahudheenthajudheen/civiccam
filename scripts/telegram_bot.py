"""
CivicCam Telegram Bot
Sends alerts for littering incidents via Telegram
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
import requests

class TelegramAlertBot:
    """Telegram bot for sending littering alerts"""
    
    def __init__(self, token: str = None, chat_id: str = None):
        """
        Initialize Telegram bot
        
        Args:
            token: Telegram bot token from @BotFather
            chat_id: Target chat/group ID
        """
        if token is None:
            try:
                # Add parent dir to path to ensure config can be imported
                import sys
                parent_dir = str(Path(__file__).parent.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
                token = TELEGRAM_BOT_TOKEN
                chat_id = TELEGRAM_CHAT_ID
            except ImportError:
                # Fallback to env vars if config.py import fails
                token = os.getenv("TELEGRAM_BOT_TOKEN", "")
                chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        self.token = token
        self.chat_id = chat_id
        self._initialized = False
        
        if not token or not chat_id:
            print("[TelegramBot] Warning: No token or chat_id provided. Alerts disabled.")
        else:
            self._initialized = True
            print("[TelegramBot] Bot initialized successfully (HTTP requests mode)")
    
    def is_configured(self) -> bool:
        """Check if bot is properly configured"""
        return bool(self.token and self.chat_id and self._initialized)
    
    def _get_url(self, endpoint: str) -> str:
        """Get the full URL for a Telegram API endpoint"""
        return f"https://api.telegram.org/bot{self.token}/{endpoint}"

    def _get_chat_ids(self) -> list:
        """Parse chat_id string into a list of individual chat IDs"""
        if not self.chat_id:
            return []
        return [cid.strip() for cid in str(self.chat_id).split(',') if cid.strip()]

    def send_alert(self, 
                   license_plate: str,
                   confidence: float,
                   location: str = "",
                   image_path: str = None,
                   incident_id: int = None) -> bool:
        """
        Send alert synchronously using standard requests API.
        This is thread-safe for WebRTC video processing blocks.
        """
        if not self.is_configured():
            print("[TelegramBot] Bot not configured. Alert not sent.")
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format message using HTML (more robust than Markdown)
            message = f"""
🚨 <b>LITTERING ALERT</b> 🚨

📋 <b>Incident Details:</b>
━━━━━━━━━━━━━━━━━━━━
🚗 <b>License Plate:</b> <code>{license_plate}</code>
📊 <b>Confidence:</b> {confidence:.1%}
📍 <b>Location:</b> {location or 'Not specified'}
🕐 <b>Time:</b> {timestamp}
🆔 <b>Incident ID:</b> #{incident_id or 'N/A'}
━━━━━━━━━━━━━━━━━━━━

⚠️ Please review and take appropriate action.
"""
            url = self._get_url("sendMessage")
            has_image = image_path and Path(image_path).exists()
            
            chat_ids = self._get_chat_ids()
            success_count = 0
            
            for cid in chat_ids:
                try:
                    # Send with image if available
                    if has_image:
                        url_photo = self._get_url("sendPhoto")
                        with open(image_path, 'rb') as photo:
                            data = {"chat_id": cid, "caption": message, "parse_mode": "HTML"}
                            files = {"photo": photo}
                            response = requests.post(url_photo, data=data, files=files, timeout=30)
                    else:
                        data = {"chat_id": cid, "text": message, "parse_mode": "HTML"}
                        response = requests.post(url, json=data, timeout=30)
                    
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        print(f"[TelegramBot] API Error for chat {cid}: {response.text}")
                except Exception as e:
                    print(f"[TelegramBot] Error sending to chat {cid}: {e}")
                
            if success_count > 0:
                print(f"[TelegramBot] Alert sent to {success_count} chats for plate: {license_plate}")
                return True
            return False
            
        except Exception as e:
            print(f"[TelegramBot] Error sending alert: {e}")
            return False

    def send_littering_alert(self,
                             license_plate: str,
                             plate_confidence: float,
                             waste_confidence: float,
                             face_confidence: float,
                             scene_image: str = None,
                             face_image: str = None,
                             plate_image: str = None,
                             waste_image: str = None,
                             incident_id: int = None,
                             location: str = "") -> bool:
        """Send comprehensive littering alert with all evidence synchronously."""
        if not self.is_configured():
            print("[TelegramBot] Bot not configured. Alert not sent.")
            return False
        
        try:
            timestamp = datetime.now().strftime("%d/%m/%Y • %H:%M:%S")
            
            # Clean, concise message using HTML
            message = f"""🚨 <b>LITTERING DETECTED</b>

🚗 <b>Plate:</b> <code>{license_plate}</code>
👤 <b>Suspect:</b> Face captured
🗑️ <b>Evidence:</b> Waste detected

📊 <b>Confidence Scores:</b>
• License Plate: {plate_confidence:.0%}
• Face Detection: {face_confidence:.0%}
• Waste Detection: {waste_confidence:.0%}

🕐 <b>Time:</b> {timestamp}
🆔 <b>Case:</b> #{incident_id or 'N/A'}

⚠️ <b>Action Required</b> - Review evidence below."""
            
            chat_ids = self._get_chat_ids()
            success_count = 0

            for cid in chat_ids:
                try:
                    # Form media group
                    media_group = []
                    files = {}
                    index = 0
                    
                    def add_media(img_path, caption):
                        nonlocal index
                        if img_path and Path(img_path).exists():
                            file_key = f"photo_{index}"
                            files[file_key] = open(img_path, 'rb')
                            
                            media_item = {
                                "type": "photo", 
                                "media": f"attach://{file_key}",
                                "parse_mode": "HTML"
                            }
                            if caption:
                                media_item["caption"] = caption
                                
                            media_group.append(media_item)
                            index += 1
                    
                    # Scene image (main photo with caption)
                    add_media(scene_image, message)
                    
                    # Face, plate, waste images
                    add_media(face_image, "👤 Suspect Face")
                    add_media(plate_image, f"🚗 Plate: {license_plate}")
                    add_media(waste_image, "🗑️ Waste Evidence")
                    
                    if len(media_group) > 1:
                        url = self._get_url("sendMediaGroup")
                        data = {"chat_id": cid, "media": json.dumps(media_group)}
                        response = requests.post(url, data=data, files=files, timeout=30)
                    elif len(media_group) == 1:
                        url = self._get_url("sendPhoto")
                        # Need to use the single object
                        file_key = list(files.keys())[0]
                        data = {"chat_id": cid, "caption": message, "parse_mode": "HTML"}
                        single_file = {"photo": files[file_key]}
                        response = requests.post(url, data=data, files=single_file, timeout=30)
                    else:
                        url = self._get_url("sendMessage")
                        data = {"chat_id": cid, "text": message, "parse_mode": "HTML"}
                        response = requests.post(url, json=data, timeout=30)
                    
                    # Ensure files are closed
                    for f in files.values():
                        f.close()
                        
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        print(f"[TelegramBot] API Error sending to {cid}: {response.text}")
                
                except Exception as inner_e:
                    print(f"[TelegramBot] Error sending to {cid}: {inner_e}")
                    
            if success_count > 0:
                print(f"[TelegramBot] Littering alert sent to {success_count} chats! Plate: {license_plate}")
                return True
            return False
            
        except Exception as e:
            print(f"[TelegramBot] Error sending littering alert setup: {e}")
            return False

    def send_test_message(self) -> bool:
        """Send a test message to verify configuration"""
        if not self.is_configured():
            print("[TelegramBot] Bot not configured")
            return False
        
        try:
            message = """
✅ <b>CivicCam Bot Test</b>

Your Telegram bot is configured correctly!
You will receive littering alerts here.

🤖 Bot Status: Active
📡 Connection: OK
"""         
            url = self._get_url("sendMessage")
            chat_ids = self._get_chat_ids()
            success_count = 0
            
            for cid in chat_ids:
                data = {"chat_id": cid, "text": message, "parse_mode": "HTML"}
                response = requests.post(url, json=data, timeout=30)
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    print(f"[TelegramBot] Test failed API error for {cid}: {response.text}")
            
            if success_count > 0:
                print(f"[TelegramBot] Test message sent successfully to {success_count} chats!")
                return True
            return False
            
        except Exception as e:
            print(f"[TelegramBot] Test failed exception: {e}")
            return False

def setup_telegram_bot():
    """Interactive setup for Telegram bot"""
    print("\n" + "="*50)
    print("Telegram Bot Setup")
    print("="*50)
    print("\nTo set up Telegram alerts:")
    print("1. Open Telegram and search for @BotFather")
    print("2. Send /newbot and follow instructions")
    print("3. Copy the API token provided")
    print("4. Create a group/channel and add your bot")
    print("5. Get the chat ID (can use @userinfobot)")
    print("\n")
    
    token = input("Enter Bot Token (or press Enter to skip): ").strip()
    chat_id = input("Enter Chat ID (or press Enter to skip): ").strip()
    
    if token and chat_id:
        # Update config file
        config_path = Path(__file__).parent.parent / "config.py"
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        content = content.replace(
            'TELEGRAM_BOT_TOKEN = ""',
            f'TELEGRAM_BOT_TOKEN = "{token}"'
        )
        content = content.replace(
            'TELEGRAM_CHAT_ID = ""',
            f'TELEGRAM_CHAT_ID = "{chat_id}"'
        )
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print("\n✓ Configuration saved!")
        
        # Test the bot
        bot = TelegramAlertBot(token, chat_id)
        if bot.send_test_message():
            print("✓ Bot is working! Check your Telegram.")
        else:
            print("✗ Test failed. Please check your credentials.")
    else:
        print("\nSetup skipped. You can configure later in config.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_telegram_bot()
    else:
        # Test with existing config
        bot = TelegramAlertBot()
        
        if bot.is_configured():
            bot.send_test_message()
        else:
            print("Bot not configured. Run: python telegram_bot.py setup")
