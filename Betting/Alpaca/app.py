import os
from dotenv import load_dotenv

load_dotenv()
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL')

print("API Key ID:", APCA_API_KEY_ID)
print("API Secret Key:", APCA_API_SECRET_KEY)
print("API Base URL:", APCA_API_BASE_URL)
