from PIL import Image
import os

folders = ["소희", "수지", "재욱", "해인"]

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") and not filename.endswith("_128.jpg"):
            image = Image.open(os.path.join(folder, filename))
            image = image.resize((128, 128))
            image.save(os.path.join(folder, filename.replace(".jpg", "_128.jpg")))