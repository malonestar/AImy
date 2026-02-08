# AImy/notify.py
import requests
import os
from config import DISCORD_WEBHOOK_URL

def send_discord_message(message: str):
    """Sends a simple text message to the Discord webhook."""
    if DISCORD_WEBHOOK_URL == "YOUR_WEBHOOK_URL_HERE":
        print("[WARN] Discord webhook URL is not set. Cannot send message.")
        return

    payload = {"content": message}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print("[INFO] Discord message sent successfully.")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to send Discord message: {e}")

def send_discord_image(image_path: str, message: str = ""):
    """Sends an image with an optional message to the Discord webhook."""
    if DISCORD_WEBHOOK_URL == "YOUR_WEBHOOK_URL_HERE":
        print("[WARN] Discord webhook URL is not set. Cannot send image.")
        return

    if not os.path.exists(image_path):
        print(f"[ERROR] Image path does not exist: {image_path}")
        return

    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f)}
            payload = {"content": message}
            response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
            response.raise_for_status()
            print("[INFO] Discord image sent successfully.")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to send Discord image: {e}")