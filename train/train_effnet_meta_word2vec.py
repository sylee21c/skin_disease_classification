#!/usr/bin/env python3
"""
EfficientNet + Metadata (Age, Sex) - Word2Vec Text Embedding Version
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
from gensim.models import KeyedVectors

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV")
parser.add_argument("--w2v-path", type=str, required=True, help="Path to Word2Vec model")
parser.add_argument(
    "--w2v-binary",
    type=int,
    default=1,
    help="1 if Word2Vec file is binary format, 0 if text format",
)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--img", type=int, default=288)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--save-dir", type=str, default="runs/ham10k_effb0_meta_w2v")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_meta_sentence(age, sex):
    if pd.isna(age):
        age = 50
    try:
        age_int = int(round(float(age)))
    except Exception:
        age_int = 50

    sex_str = str(sex).lower()
    if "male" in sex_str or sex_str.startswith("m"):
        sex_word = "male"
    elif "female" in sex_str or sex_str.startswith("f"):
        sex_word = "female"
    else:
        sex_word = "unknown"

    sentence = f"skin lesion of a {age_int} year old {sex_word} patient"
    return sentence


class HAM10000WithMetaText(Dataset):
    def __init__(self, image_dir, metadata_df, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.metadata = metadata_df.copy()

        self.samples = []
        for class_name in os.listdir(image_dir):
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = 0 if class_name.lower() == "benign" else 1
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_dir, img_name)
                    image_id = os.path.splitext(img_name)[0]
                    self.samples.append((img_path, class_idx, image_id))

        print(f"[Word2Vec] Loaded {len(self.samples)} images from {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, image_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        meta_row = self.metadata[self.metadata["isic_id"] == image_id]
        if len(meta_row) > 0:
            age = meta_row["age_approx"].values[0]
            sex = meta_row["sex"].values[0]
        else:
            age = 50
            sex = "female"

        meta_text = build_meta_sentence(age, sex)
        label = torch.tensor(label, dtype=torch.long)
        return image, meta_text, label


def encode_metadata_w2v(text_list, w2v_model, dim):
    vectors = []
    for text in text_list:
        tokens = str(text).lower().split()
        token_vecs = []
        for t in tokens:
            if t in w2v_model.key_to_index:
                token_vecs.append(w2v_model[t])
        if len(token_vecs) == 0:
            vectors.append(np.zeros(dim, dtype=np.float32))
        else:
            vectors.append(np.mean(token_vecs, axis=0).astype(np.float32))
    vectors = np.stack(vectors, axis=0)
    return torch.from_numpy(vectors)


class EfficientNetWithMetadata(nn.Module):
    def __init__(self, num_classes=2, metadata_dim=300):
        super().__init__()
        self.effnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, metadata):
        img_feat = self.effnet(img)
        meta_feat = self.metadata_fc(metadata)
        combined = torch.cat([img_feat, meta_feat], dim=1)
        out = self.classifier(combined)
        return out


train_tf = transforms.Compose(
    [
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)
val_tf = transforms.Compose(
    [
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

print(f"Loading metadata from {args.metadata}")
metadata_df = pd.read_csv(args.metadata)
print(f"Metadata shape: {metadata_df.shape}")
print(f"Columns: {metadata_df.columns.tolist()}")

print(f"Loading Word2Vec model from {args.w2v_path}")
w2v = KeyedVectors.load_word2vec_format(args.w2v_path, binary=bool(args.w2v_binary))
w2v_dim = w2v.vector_size
print(f"Word2Vec dim: {w2v_dim}")

train_path = os.path.join(args.data, "train")
val_path = os.path.join(args.data, "val")

train_ds = HAM10000WithMetaText(train_path, metadata_df, transform=train_tf)
val_ds = HAM10000WithMetaText(val_path, metadata_df, transform=val_tf)

train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

model = EfficientNetWithMetadata(num_classes=2, metadata_dim=w2v_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

os.makedirs(args.save_dir, exist_ok=True)
best_acc = 0.0

print("\n" + "=" * 60)
print("EfficientNet + Word2Vec Metadata Training")
print("=" * 60 + "\n")

for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0

    for imgs, meta_texts, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        meta_vec = encode_metadata_w2v(meta_texts, w2v, w2v_dim)
        meta_vec = meta_vec.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs, meta_vec)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, meta_texts, labels in val_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            meta_vec = encode_metadata_w2v(meta_texts, w2v, w2v_dim)
            meta_vec = meta_vec.to(device, non_blocking=True)

            logits = model(imgs, meta_vec)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(train_dl)
    print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        save_path = os.path.join(args.save_dir, "effb0_meta_w2v_best.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  ✓ New best model saved to {save_path} (Acc: {acc:.4f})")

print("\n" + "=" * 60)
print("Training Complete (Word2Vec)")
print(f"Best Validation Accuracy: {best_acc:.4f}")
print("=" * 60)
