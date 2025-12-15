import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import argparse
import os
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--img", type=int, default=288)
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ 데이터셋 로드 (먼저 클래스 분포 확인)
train_path = os.path.join(args.data, "train")
val_path = os.path.join(args.data, "val")
train_ds_temp = datasets.ImageFolder(train_path)

class_counts = np.bincount(train_ds_temp.targets)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = torch.FloatTensor(class_weights).to(device)
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")

# 2️⃣ 클래스별 증강 설정
train_tf = transforms.Compose([
    transforms.Resize((args.img, args.img)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

val_tf = transforms.Compose([
    transforms.Resize((args.img, args.img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

train_ds = datasets.ImageFolder(train_path, transform=train_tf)
val_ds = datasets.ImageFolder(val_path, transform=val_tf)

train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

num_classes = len(train_ds.classes)
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes).to(device)

# 3️⃣ class-weighted loss 적용
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_acc = 0.0
os.makedirs("runs/ham10k_effb0_weighted", exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            _, pred_cls = torch.max(preds, 1)
            correct += (pred_cls == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_dl):.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "runs/ham10k_effb0_weighted/effb0_best.pt")
        print("Best model updated.")

print(f"Training complete. Best Val Acc: {best_acc:.4f}")

