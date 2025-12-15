#!/usr/bin/env python3
"""
EfficientNet + Metadata (Age, Sex)
이미지와 메타데이터를 함께 사용
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--img", type=int, default=288)
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Custom Dataset with Metadata
class HAM10000WithMetadata(Dataset):
    def __init__(self, image_dir, metadata_df, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.metadata = metadata_df
        
        # 이미지 파일 목록
        self.samples = []
        for class_name in os.listdir(image_dir):
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            class_idx = 0 if class_name.lower() == 'benign' else 1
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    # 파일명에서 ISIC ID 추출 (ISIC_xxxxxxx)
                    image_id = os.path.splitext(img_name)[0]
                    self.samples.append((img_path, class_idx, image_id))
        
        print(f"Loaded {len(self.samples)} images from {image_dir}")
        
        # 나이 구간화 (10년 단위) - 컬럼명 수정!
        self.metadata['age_bin'] = pd.cut(self.metadata['age_approx'], 
                                           bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                           labels=False, 
                                           include_lowest=True)
        self.metadata['age_bin'] = self.metadata['age_bin'].fillna(5)  # 기본값 50대
        
        # 성별 인코딩
        self.metadata['sex_encoded'] = (self.metadata['sex'] == 'male').astype(int)
        self.metadata['sex_encoded'] = self.metadata['sex_encoded'].fillna(0)  # 기본값 female
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, image_id = self.samples[idx]
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 메타데이터 가져오기 - 컬럼명 수정!
        meta_row = self.metadata[self.metadata['isic_id'] == image_id]
        
        if len(meta_row) > 0:
            age_bin = int(meta_row['age_bin'].values[0])
            sex = int(meta_row['sex_encoded'].values[0])
        else:
            # 메타데이터 없으면 기본값
            age_bin = 5  # 50대
            sex = 0  # female
        
        # 메타데이터를 one-hot encoding
        age_onehot = torch.zeros(10)
        age_onehot[age_bin] = 1
        
        sex_onehot = torch.zeros(2)
        sex_onehot[sex] = 1
        
        metadata = torch.cat([age_onehot, sex_onehot])  # 12차원
        
        return image, metadata, label

# Model with Metadata
class EfficientNetWithMetadata(nn.Module):
    def __init__(self, num_classes=2, metadata_dim=12):
        super().__init__()
        
        # 이미지 feature extractor
        self.effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # 메타데이터 처리
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 통합 classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, img, metadata):
        img_feat = self.effnet(img)  # [batch, 1280]
        meta_feat = self.metadata_fc(metadata)  # [batch, 128]
        
        combined = torch.cat([img_feat, meta_feat], dim=1)  # [batch, 1408]
        output = self.classifier(combined)
        
        return output

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

# Load metadata
print(f"Loading metadata from {args.metadata}")
metadata_df = pd.read_csv(args.metadata)
print(f"Metadata shape: {metadata_df.shape}")
print(f"Columns: {metadata_df.columns.tolist()}")
print(f"Sample data:")
print(metadata_df[['isic_id', 'age_approx', 'sex']].head())

# Datasets
train_path = os.path.join(args.data, "train")
val_path = os.path.join(args.data, "val")

train_ds = HAM10000WithMetadata(train_path, metadata_df, transform=train_tf)
val_ds = HAM10000WithMetadata(val_path, metadata_df, transform=val_tf)

train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

# Model
model = EfficientNetWithMetadata(num_classes=2, metadata_dim=12).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Training
best_acc = 0.0
save_dir = "runs/ham10k_effb0_with_metadata"
os.makedirs(save_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"EfficientNet + Metadata Training")
print(f"{'='*60}\n")

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    
    for imgs, metadata, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
        imgs = imgs.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        preds = model(imgs, metadata)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, metadata, labels in val_dl:
            imgs = imgs.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            
            preds = model(imgs, metadata)
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
