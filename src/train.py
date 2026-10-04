import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import FaceDataset
from model import FaceCNN


dataset = FaceDataset()
print("전체 이미지 개수:", len(dataset))

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)
print("학습 이미지 개수:", len(train_dataset))
print("테스트 이미지 개수:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)


model = FaceCNN()


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 20

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )


torch.save(model.state_dict(), "face_cnn.pth")

print("학습 완료!")
print("모델 저장 완료: face_cnn.pth")