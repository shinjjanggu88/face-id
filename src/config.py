import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FOLDER_NAMES = ["소희", "수지", "재욱", "해인"]

FOLDERS = [
    os.path.join(BASE_DIR, folder)
    for folder in FOLDER_NAMES
]