#!/usr/bin/env python3
"""
EfficientNet + Metadata (OneHot / Word2Vec / Word2Vec_2 / CLIP Text) 평가 스크립트

- 같은 val 세트에 대해
    1) One-Hot 메타데이터 모델
    2) Word2Vec 메타데이터 모델
    3) Word2Vec_2 메타데이터 모델 (두 번째 실험)
    4) CLIP Text 메타데이터 모델
  의 성능을 비교하고, 터미널에 표 형식으로 출력한다.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
from gensim.models import KeyedVectors
import open_clip
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

# -----------------------------------------------------------
#   Argument Parser
# -----------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="root dir (train/ val/)")
parser.add_argument("--metadata", type=str, required=True, help="metadata CSV path")
parser.add_argument("--img", type=int, default=288)
parser.add_argument("--batch", type=int, default=32)

parser.add_argument("--model-onehot", type=str, required=True)
parser.add_argument("--model-w2v", type=str, required=True)
parser.add_argument("--model-w2v2", type=str, required=True)  # 두 번째 Word2Vec 모델
parser.add_argument("--model-clip", type=str, required=True)

parser.add_argument("--w2v-path", type=str, required=True)
parser.add_argument(
    "--w2v-binary",
    type=int,
    default=1,
    help="1 if Word2Vec file is binary format, 0 if text format",
)

parser.add_argument("--clip-model-name", type=str, default="ViT-B-32")
parser.add_argument("--clip-pretrained", type=str, default="openai")

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------
#   Helper: 메타 문장 만들기
# -----------------------------------------------------------

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


# -----------------------------------------------------------
#   Dataset: val 용 (이미지 + onehot 메타 + 텍스트 메타)
# -----------------------------------------------------------

class HAM10000EvalDataset(Dataset):
    """
    하나의 Dataset에서
      - image
      - metadata_onehot (age_bin + sex)
      - meta_text (Word2Vec / CLIP에서 공통으로 사용)
      - label
    을 모두 반환
    """

    def __init__(self, image_dir, metadata_df, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.metadata = metadata_df.copy()

        # age_bin, sex_encoded 미리 계산
        self.metadata["age_bin"] = pd.cut(
            self.metadata["age_approx"],
            bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            labels=False,
            include_lowest=True,
        )
        self.metadata["age_bin"] = self.metadata["age_bin"].fillna(5).astype(int)
        self.metadata["sex_encoded"] = (self.metadata["sex"] == "male").astype(int)
        self.metadata["sex_encoded"] = self.metadata["sex_encoded"].fillna(0).astype(int)

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

        print(f"[Eval] Loaded {len(self.samples)} images from {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, image_id = self.samples[idx]

        # 이미지
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # 메타데이터
        meta_row = self.metadata[self.metadata["isic_id"] == image_id]
        if len(meta_row) > 0:
            age_bin = int(meta_row["age_bin"].values[0])
            sex_enc = int(meta_row["sex_encoded"].values[0])
            age = meta_row["age_approx"].values[0]
            sex = meta_row["sex"].values[0]
        else:
            age_bin = 5
            sex_enc = 0
            age = 50
            sex = "female"

        # one-hot (age, sex)
        age_onehot = torch.zeros(10, dtype=torch.float32)
        age_onehot[age_bin] = 1.0
        sex_onehot = torch.zeros(2, dtype=torch.float32)
        sex_onehot[sex_enc] = 1.0
        metadata_onehot = torch.cat([age_onehot, sex_onehot], dim=0)  # 12차원

        # 텍스트 메타 (Word2Vec/CLIP 공용)
        meta_text = build_meta_sentence(age, sex)

        label = torch.tensor(label, dtype=torch.long)
        return image, metadata_onehot, meta_text, label


# -----------------------------------------------------------
#   Helper: Word2Vec / CLIP 인코딩
# -----------------------------------------------------------

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


def encode_metadata_clip(text_list, clip_model, tokenizer, device):
    with torch.no_grad():
        tokens = tokenizer(text_list)
        tokens = tokens.to(device)
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.float()


# -----------------------------------------------------------
#   EfficientNet + Metadata 모델 정의 (평가용)
# -----------------------------------------------------------

class EfficientNetWithMetadata(nn.Module):
    def __init__(self, num_classes=2, metadata_dim=12):
        super().__init__()
        self.effnet = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0
        )
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


# -----------------------------------------------------------
#   데이터 / 모델 로드
# -----------------------------------------------------------

val_tf = transforms.Compose(
    [
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

print(f"[Info] Loading metadata from {args.metadata}")
metadata_df = pd.read_csv(args.metadata)
print(f"  Metadata shape: {metadata_df.shape}")

val_path = os.path.join(args.data, "val")
val_ds = HAM10000EvalDataset(val_path, metadata_df, transform=val_tf)
val_dl = DataLoader(
    val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True
)

print(f"[Info] Loading Word2Vec model from {args.w2v_path}")
w2v = KeyedVectors.load_word2vec_format(args.w2v_path, binary=bool(args.w2v_binary))
w2v_dim = w2v.vector_size
print(f"  Word2Vec dim: {w2v_dim}")

print(f"[Info] Loading CLIP text encoder: {args.clip_model_name}, pretrained={args.clip_pretrained}")
clip_model, _, _ = open_clip.create_model_and_transforms(
    args.clip_model_name, pretrained=args.clip_pretrained
)
tokenizer = open_clip.get_tokenizer(args.clip_model_name)
clip_model = clip_model.to(device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
clip_text_dim = clip_model.text_projection.shape[1]
print(f"  CLIP text dim: {clip_text_dim}")

# 모델 4개 생성
model_onehot = EfficientNetWithMetadata(num_classes=2, metadata_dim=12).to(device)
model_w2v = EfficientNetWithMetadata(num_classes=2, metadata_dim=w2v_dim).to(device)
model_w2v2 = EfficientNetWithMetadata(num_classes=2, metadata_dim=w2v_dim).to(device)
model_clip = EfficientNetWithMetadata(num_classes=2, metadata_dim=clip_text_dim).to(device)

print("\n[Info] Loading model weights:")
print(f"  OneHot   : {args.model-onehot if False else args.model_onehot}")
print(f"  Word2Vec 1: {args.model_w2v}")
print(f"  Word2Vec 2: {args.model_w2v2}")
print(f"  CLIP     : {args.model_clip}")

state_onehot = torch.load(args.model_onehot, map_location=device)
state_w2v = torch.load(args.model_w2v, map_location=device)
state_w2v2 = torch.load(args.model_w2v2, map_location=device)
state_clip = torch.load(args.model_clip, map_location=device)

model_onehot.load_state_dict(state_onehot)
model_w2v.load_state_dict(state_w2v)
model_w2v2.load_state_dict(state_w2v2)
model_clip.load_state_dict(state_clip)

models = {
    "onehot": model_onehot,
    "w2v": model_w2v,
    "w2v2": model_w2v2,
    "clip": model_clip,
}

pretty_name = {
    "onehot": "One-Hot (원핫)",
    "w2v": "Word2Vec-1",
    "w2v2": "Word2Vec-2",
    "clip": "CLIP Text",
}

results = {}


# -----------------------------------------------------------
#   Evaluation Loop
# -----------------------------------------------------------

for name, model in models.items():
    print("\n" + "=" * 70)
    print(f"[ {pretty_name[name]} ] 평가 중...".center(70))
    print("=" * 70)

    model.eval()
    all_labels = []
    all_probs = []  # malignant(1) 확률

    with torch.no_grad():
        for imgs, metadata_onehot, meta_texts, labels in val_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if name == "onehot":
                meta_vec = metadata_onehot.to(device, non_blocking=True)
            elif name in ("w2v", "w2v2"):
                meta_vec = encode_metadata_w2v(meta_texts, w2v, w2v_dim)
                meta_vec = meta_vec.to(device, non_blocking=True)
            else:  # clip
                meta_vec = encode_metadata_clip(meta_texts, clip_model, tokenizer, device)
                meta_vec = meta_vec.to(device, non_blocking=True)

            logits = model(imgs, meta_vec)
            probs = torch.softmax(logits, dim=1)[:, 1]  # malignant 확률

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    acc = (preds == all_labels).mean()
    f1 = f1_score(all_labels, preds)
    prec = precision_score(all_labels, preds)
    rec = recall_score(all_labels, preds)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(all_labels, preds)
    cls_report = classification_report(
        all_labels, preds, target_names=["Benign", "Malignant"]
    )

    results[name] = {
        "acc": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc,
        "cm": cm,
        "report": cls_report,
    }

    # 간단 출력
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\n[Classification Report]")
    print(cls_report)
    print("[Confusion Matrix] (rows: true, cols: pred)")
    print(cm)


# -----------------------------------------------------------
#   Pretty Print: 전 모델 성능 비교 + 최고 모델 요약
# -----------------------------------------------------------

print("\n" + "=" * 70)
print("전   모   델   성   능   비   교".center(70))
print("=" * 70)

header = (
    f"{'모델':15s} | "
    f"{'Acc':7s} | "
    f"{'F1':7s} | "
    f"{'Precision':9s} | "
    f"{'Recall':7s} | "
    f"{'AUC':7s}"
)
print(header)
print("-" * len(header))

for name in ["onehot", "w2v", "w2v2", "clip"]:
    r = results[name]
    line = (
        f"{pretty_name[name]:15s} | "
        f"{r['acc']:.4f} | "
        f"{r['f1']:.4f} | "
        f"{r['precision']:.4f}   | "
        f"{r['recall']:.4f} | "
        f"{r['roc_auc']:.4f}"
    )
    print(line)

# 최고 값 찾기
model_keys = ["onehot", "w2v", "w2v2", "clip"]

best_acc_name = max(model_keys, key=lambda n: results[n]["acc"])
best_f1_name = max(model_keys, key=lambda n: results[n]["f1"])
best_auc_name = max(model_keys, key=lambda n: results[n]["roc_auc"])

print("\n" + "=" * 70)
print("최   성   능   모   델".center(70))
print("=" * 70)
print(
    f"최고 Accuracy : {pretty_name[best_acc_name]} "
    f"({results[best_acc_name]['acc']:.4f})"
)
print(
    f"최고 F1-Score : {pretty_name[best_f1_name]} "
    f"({results[best_f1_name]['f1']:.4f})"
)
print(
    f"최고 AUC      : {pretty_name[best_auc_name]} "
    f"({results[best_auc_name]['roc_auc']:.4f})"
)
print("-" * 70)
print("평가 완료")
print("-" * 70)
