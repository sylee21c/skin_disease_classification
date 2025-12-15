#!/usr/bin/env python3
"""
Mixup Data Augmentation - Same Label Only
동일 레이블끼리만 섞기 (Benign끼리, Malignant끼리)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import argparse
import os
from tqdm import tqdm
from collections import Counter
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--img", type=int, default=288)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--mixup-alpha", type=float, default=0.2, help="Mixup alpha parameter")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def mixup_data_same_label(x, y, alpha=0.2):
    """
    동일 레이블끼리만 Mixup 수행
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    
    # 각 샘플마다 같은 레이블을 가진 다른 샘플과 섞기
    mixed_x = x.clone()
    for i in range(batch_size):
        # 같은 레이블을 가진 인덱스 찾기
        same_label_indices = (y == y[i]).nonzero(as_tuple=True)[0]
        
        # 자기 자신 제외
        same_label_indices = same_label_indices[same_label_indices != i]
        
        if len(same_label_indices) > 0:
            # 랜덤하게 하나 선택
            j = same_label_indices[torch.randint(len(same_label_indices), (1,))].item()
            mixed_x[i] = lam * x[i] + (1 - lam) * x[j]
    
    return mixed_x, y  # 레이블은 그대로

# Data transforms
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

# Load datasets
train_path = os.path.join(args.data, "train")
val_path = os.path.join(args.data, "val")

train_ds = datasets.ImageFolder(train_path, transform=train_tf)
val_ds = datasets.ImageFolder(val_path, transform=val_tf)

# Class weights
class_counts = np.bincount([label for _, label in train_ds.samples])
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = torch.FloatTensor(class_weights).to(device)

print(f"\n{'='*60}")
print(f"Mixup Augmentation - Same Label Only")
print(f"{'='*60}")
print(f"클래스: {train_ds.classes}")
print(f"클래스별 개수: {class_counts}")
print(f"클래스 가중치: {class_weights.cpu().numpy()}")
print(f"Mixup Alpha: {args.mixup_alpha}")
print(f"{'='*60}\n")

train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

# Model
num_classes = len(train_ds.classes)
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Training
best_acc = 0.0
save_dir = "runs/ham10k_effb0_mixup_same"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    
    for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Apply Mixup (same label only)
        imgs, labels = mixup_data_same_label(imgs, labels, alpha=args.mixup_alpha)
        
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
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
    avg_loss = total_loss / len(train_dl)
    
    print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{save_dir}/effb0_best.pt")
        print(f"  ✓ Best model saved! (Acc: {acc:.4f})")

print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"Best Validation Accuracy: {best_acc:.4f}")
print(f"Model saved to: {save_dir}/effb0_best.pt")
print(f"{'='*60}")
