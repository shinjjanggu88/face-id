import torch
from PIL import Image
from torchvision import transforms

from config import CLASS_NAMES
from model import FaceCNN

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

model = FaceCNN()
model.load_state_dict(torch.load("face_cnn.pth"))
model.eval()

image_path = input("테스트할 이미지 경로를 입력하세요: ")

image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1)

print("예측 결과:", CLASS_NAMES[prediction.item()])