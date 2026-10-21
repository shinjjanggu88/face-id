
from pathlib import Path
import sys
import io

import torch
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException

# src 폴더의 기존 코드 가져오기
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SRC_DIR))

from config import CLASS_NAMES
from model import FaceCNN

app = FastAPI(title="CNN Face ID API")

MODEL_PATH = BASE_DIR / "face_cnn.pth"
DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

model = FaceCNN()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.to(DEVICE)
model.eval()


@app.get("/")
def home():
    return {"message": "CNN Face ID API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail="JPG, PNG, WEBP 이미지 파일만 업로드할 수 있습니다."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="이미지 파일을 읽을 수 없습니다."
        )

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    return {
        "filename": file.filename,
        "prediction": CLASS_NAMES[prediction],
        "confidence": round(probabilities[0, prediction].item(), 4)
    }
