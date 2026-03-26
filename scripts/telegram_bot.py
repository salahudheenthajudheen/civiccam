"""
CivicCam Telegram Bot
Sends alerts for littering incidents via Telegram
"""

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
import io


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
            from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
            token = TELEGRAM_BOT_TOKEN
            chat_id = TELEGRAM_CHAT_ID
        
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        self._initialized = False
        
        if not token:
            print("[TelegramBot] Warning: No token provided. Alerts disabled.")
        else:
            self._init_bot()
    
    def _init_bot(self):
        """Initialize the bot with increased timeout"""
        try:
            from telegram import Bot
            from telegram.request import HTTPXRequest
            
            # Increase timeout for slow connections
            request = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0, write_timeout=30.0)
            self.bot = Bot(token=self.token, request=request)
            self._initialized = True
            print("[TelegramBot] Bot initialized successfully (30s timeout)")
        except ImportError:
            print("[TelegramBot] python-telegram-bot not installed. Run: pip install python-telegram-bot")
        except Exception as e:
            print(f"[TelegramBot] Error initializing bot: {e}")
    
    def is_configured(self) -> bool:
        """Check if bot is properly configured"""
        return bool(self.token and self.chat_id and self._initialized)
    
    async def send_alert_async(self, 
                               license_plate: str,
                               confidence: float,
                               location: str = "",
                               image_path: str = None,
                               incident_id: int = None) -> bool:
        """
        Send alert asynchronously
        
        Args:
            license_plate: Detected license plate
            confidence: Detection confidence
            location: Location description
            image_path: Path to incident image
            incident_id: Database incident ID
            
        Returns:
            True if sent successfully
        """
        if not self.is_configured():
            print("[TelegramBot] Bot not configured. Alert not sent.")
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format message
            message = f"""
🚨 **LITTERING ALERT** 🚨

📋 **Incident Details:**
━━━━━━━━━━━━━━━━━━━━
🚗 **License Plate:** `{license_plate}`
📊 **Confidence:** {confidence:.1%}
📍 **Location:** {location or 'Not specified'}
🕐 **Time:** {timestamp}
🆔 **Incident ID:** #{incident_id or 'N/A'}
━━━━━━━━━━━━━━━━━━━━

⚠️ Please review and take appropriate action.
"""
            
            # Send with image if available
            if image_path and Path(image_path).exists():
                with open(image_path, 'rb') as photo:
                    await self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=photo,
                        caption=message,
                        parse_mode='Markdown'
                    )
            else:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
            
            print(f"[TelegramBot] Alert sent for plate: {license_plate}")
            return True
            
        except Exception as e:
            print(f"[TelegramBot] Error sending alert: {e}")
            return False
    
    def send_alert(self, 
                   license_plate: str,
                   confidence: float,
                   location: str = "",
                   image_path: str = None,
                   incident_id: int = None) -> bool:
        """
        Send alert synchronously (wrapper for async method)
        
        Args:
            license_plate: Detected license plate
            confidence: Detection confidence
            location: Location description
            image_path: Path to incident image
            incident_id: Database incident ID
            
        Returns:
            True if sent successfully
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.send_alert_async(
                license_plate=license_plate,
                confidence=confidence,
                location=location,
                image_path=image_path,
                incident_id=incident_id
            )
        )
    
    async def send_littering_alert_async(self,
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
        """
        Send comprehensive littering alert with all evidence
        """
        if not self.is_configured():
            print("[TelegramBot] Bot not configured. Alert not sent.")
            return False
        
        try:
            from telegram import InputMediaPhoto
            timestamp = datetime.now().strftime("%d/%m/%Y • %H:%M:%S")
            
            # Clean, concise message
            message = f"""🚨 *LITTERING DETECTED*

🚗 *Plate:* `{license_plate}`
👤 *Suspect:* Face captured
🗑️ *Evidence:* Waste detected

📊 *Confidence Scores:*
• License Plate: {plate_confidence:.0%}
• Face Detection: {face_confidence:.0%}
• Waste Detection: {waste_confidence:.0%}

🕐 *Time:* {timestamp}
🆔 *Case:* #{incident_id or 'N/A'}

⚠️ *Action Required* - Review evidence below."""
            
            # Collect available images
            media_group = []
            
            # Scene image (main photo with caption)
            if scene_image and Path(scene_image).exists():
                with open(scene_image, 'rb') as f:
                    scene_bytes = f.read()
                media_group.append(InputMediaPhoto(
                    media=scene_bytes,
                    caption=message,
                    parse_mode='Markdown'
                ))
            
            # Face image
            if face_image and Path(face_image).exists():
                with open(face_image, 'rb') as f:
                    face_bytes = f.read()
                media_group.append(InputMediaPhoto(
                    media=face_bytes,
                    caption="👤 Suspect Face"
                ))
            
            # License plate image
            if plate_image and Path(plate_image).exists():
                with open(plate_image, 'rb') as f:
                    plate_bytes = f.read()
                media_group.append(InputMediaPhoto(
                    media=plate_bytes,
                    caption=f"🚗 Plate: {license_plate}"
                ))
            
            # Waste image
            if waste_image and Path(waste_image).exists():
                with open(waste_image, 'rb') as f:
                    waste_bytes = f.read()
                media_group.append(InputMediaPhoto(
                    media=waste_bytes,
                    caption="🗑️ Waste Evidence"
                ))
            
            # Send media group if we have multiple images
            if len(media_group) > 1:
                await self.bot.send_media_group(
                    chat_id=self.chat_id,
                    media=media_group
                )
            elif len(media_group) == 1:
                # Just send single photo
                if scene_image and Path(scene_image).exists():
                    with open(scene_image, 'rb') as photo:
                        await self.bot.send_photo(
                            chat_id=self.chat_id,
                            photo=photo,
                            caption=message,
                            parse_mode='Markdown'
                        )
            else:
                # No images, just send text
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
            
            print(f"[TelegramBot] Littering alert sent! Plate: {license_plate}")
            return True
            
        except Exception as e:
            print(f"[TelegramBot] Error sending littering alert: {e}")
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
        """Send littering alert synchronously"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.send_littering_alert_async(
                license_plate=license_plate,
                plate_confidence=plate_confidence,
                waste_confidence=waste_confidence,
                face_confidence=face_confidence,
                scene_image=scene_image,
                face_image=face_image,
                plate_image=plate_image,
                waste_image=waste_image,
                incident_id=incident_id,
                location=location
            )
        )
    
    async def send_test_message_async(self) -> bool:
        """Send a test message to verify configuration"""
        if not self.is_configured():
            print("[TelegramBot] Bot not configured")
            return False
        
        try:
            message = """
✅ **CivicCam Bot Test**

Your Telegram bot is configured correctly!
You will receive littering alerts here.

🤖 Bot Status: Active
📡 Connection: OK
"""
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            print("[TelegramBot] Test message sent successfully!")
            return True
        except Exception as e:
            print(f"[TelegramBot] Test failed: {e}")
            return False
    
    def send_test_message(self) -> bool:
        """Send test message synchronously"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.send_test_message_async())


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
