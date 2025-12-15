#!/usr/bin/env python3
"""
CutMix Data Augmentation
이미지 패치를 잘라서 다른 이미지에 붙이기
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
parser.add_argument("--cutmix-alpha", type=float, default=1.0, help="CutMix alpha parameter")
parser.add_argument("--cutmix-prob", type=float, default=0.5, help="Probability of applying CutMix")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def rand_bbox(size, lam):
    """
    Random bounding box 생성
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, num_classes=2):
    """
    CutMix augmentation
    이미지의 일부 영역을 다른 이미지의 영역으로 대체
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    # Generate random box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # Cut and mix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match the actual area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    # Mix labels
    y_a = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    y_b = torch.nn.functional.one_hot(y[index], num_classes=num_classes).float()
    mixed_y = lam * y_a + (1 - lam) * y_b
    
    return x, mixed_y

def cutmix_criterion(pred, y_mixed):
    """
    Mixed label을 위한 손실 함수
    """
    return -torch.mean(torch.sum(y_mixed * torch.log_softmax(pred, dim=1), dim=1))

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

# Class info
class_counts = np.bincount([label for _, label in train_ds.samples])

print(f"\n{'='*60}")
print(f"CutMix Augmentation")
print(f"{'='*60}")
print(f"클래스: {train_ds.classes}")
print(f"클래스별 개수: {class_counts}")
print(f"CutMix Alpha: {args.cutmix_alpha}")
print(f"CutMix Probability: {args.cutmix_prob}")
print(f"{'='*60}\n")

train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

# Model
num_classes = len(train_ds.classes)
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# For validation
val_criterion = nn.CrossEntropyLoss()

# Training
best_acc = 0.0
save_dir = "runs/ham10k_effb0_cutmix"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    
    for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Apply CutMix with probability
        r = np.random.rand(1)
        if r < args.cutmix_prob:
            imgs, mixed_labels = cutmix_data(imgs, labels, 
                                              alpha=args.cutmix_alpha, 
                                              num_classes=num_classes)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = cutmix_criterion(preds, mixed_labels)
        else:
            # Regular training
            optimizer.zero_grad()
            preds = model(imgs)
            loss = val_criterion(preds, labels)
        
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
