import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from talkbuddy.config import settings

def test_settings():
    print("Settings:")
    print(f"OPENAI_API_KEY: {settings.OPENAI_API_KEY}")
    print(f"DEBUG: {settings.DEBUG}")
    print(f"ROOMS_FILE: {settings.ROOMS_FILE}")

if __name__ == "__main__":
    test_settings()