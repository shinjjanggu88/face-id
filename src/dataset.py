from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

folders = ["소희", "수지", "재욱", "해인"]


class FaceDataset(Dataset):
    def __init__(self):
        self.images = []
        self.labels = []

        for label, folder in enumerate(folders):
            for filename in os.listdir(folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    path = os.path.join(folder, filename)
                    self.images.append(path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = transform(image)

        label = self.labels[index]

        return image, label


dataset = FaceDataset()

print("클래스:", folders)
print("이미지 개수:", len(dataset))