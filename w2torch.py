import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用GPU
    print("Using CUDA for training")
else:
    device = torch.device("cpu")  # 使用CPU
    print("Using CPU for training")

# 自定義資料集
class FoxDataset(Dataset):
    def __init__(self, cat_dir, dog_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        for label, folder in enumerate([cat_dir, dog_dir]):
            for file in os.listdir(folder):
                if file.endswith(".jpg"):
                    img = Image.open(os.path.join(folder, file)).convert('RGB')
                    self.images.append(img)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 資料路徑
catimg = 'C:/Users/stars/Desktop/w2_final/testtrain/training_set/cats'
dogimg = 'C:/Users/stars/Desktop/w2_final/testtrain/training_set/dogs'

# 定義資料增強與正規化
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 建立資料集與 DataLoader
dataset = FoxDataset(catimg, dogimg, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 資料視覺化
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 建立模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 移除 `activation` 參數
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # 在 forward 中手動加入激活函數
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 30 * 30)  # 展平成一維
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN()
model = CNN().to(device)

# 定義損失函數與優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
for epoch in range(5):  # 訓練 5 個 epoch
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 移動到 GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 次迭代打印一次
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")

# 評估模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # 移動到 GPU
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the test images: {100 * correct / total:.2f}%")
