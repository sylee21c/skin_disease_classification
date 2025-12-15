#!/usr/bin/env python3
"""
CLIP backbone + metadata classifier (HAM10000 binary)

기존 EfficientNet+metadata 학습 스크립트를 CLIP 기반으로 변경한 버전.

- CLIP 이미지 인코더를 backbone으로 사용
- 이미지 임베딩 + 메타데이터(age, sex)를 concat해서 MLP classifier 학습
- 하나의 CLIP 모델에 대해 다음 4가지 설정을 선택해서 학습 가능:
    1) image          : 이미지만
    2) image_sex      : 이미지 + 성별
    3) image_age      : 이미지 + 나이
    4) image_age_sex  : 이미지 + 나이 + 성별

사용 예:
  python -u train_effnet_with_metadata.py \
    --data data/processed/ham10000_bin \
    --metadata ham10k_meta_for_effnet.csv \
    --meta-mode image_age_sex \
    --epochs 20 \
    --batch 32 \
    --model-name "ViT-B/32"
"""

import os
import csv
import argparse
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tqdm import tqdm

import clip  # openai/CLIP 패키지 (https://github.com/openai/CLIP)


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------

class HamMetaCLIPDataset(Dataset):
    """
    HAM10000 + metadata용 Dataset.

    CSV 컬럼 (기본값):
      - image_id : 파일 이름 (예: ISIC_XXXXXXX.jpg)
      - target   : 0 or 1 (benign / malignant)
      - sex      : 'male', 'female', 기타
      - age      : 정수 나이 (or float)
      - split    : 'train' / 'val'

    이미지 경로 가정:
      <data_root>/<split>/<image_id>
    """

    def __init__(
        self,
        data_root: str,
        metadata_csv: str,
        split: str,
        preprocess,
        image_col: str = "image_id",
        label_col: str = "target",
        sex_col: str = "sex",
        age_col: str = "age",
        split_col: str = "split",
    ):
        self.data_root = data_root
        self.preprocess = preprocess
        self.samples: List[Tuple[str, int, float, int]] = []

        with open(metadata_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = row.get(split_col, "").strip().lower()
                if row_split != split.lower():
                    continue

                img_name = row.get(image_col)
                if not img_name:
                    continue

                label = self._parse_label(row.get(label_col))
                sex_id = self._parse_sex(row.get(sex_col))
                age = self._parse_age(row.get(age_col))

                # 기본 구조: <data_root>/<split>/<image_id>
                img_path = os.path.join(data_root, split, img_name)
                if not os.path.isfile(img_path):
                    # 필요하면 여기를 바꿔서 자신의 폴더 구조에 맞추면 됨.
                    # 예: img_path = os.path.join(data_root, img_name)
                    # print(f"[WARN] image not found: {img_path}")
                    continue

                self.samples.append((img_path, label, age, sex_id))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}' in '{metadata_csv}'. "
                "CSV 컬럼 이름, split 값, data_root 경로를 확인해봐."
            )

    @staticmethod
    def _parse_label(raw_label) -> int:
        if raw_label is None:
            return 0
        s = str(raw_label).strip().lower()
        if s.isdigit():
            return int(s)
        if "malig" in s:
            return 1
        if "benign" in s:
            return 0
        return 0

    @staticmethod
    def _parse_sex(raw_sex) -> int:
        """
        0: male, 1: female, 2: unknown
        """
        if raw_sex is None:
            return 2
        s = str(raw_sex).strip().lower()
        if s in ["male", "m", "man"]:
            return 0
        if s in ["female", "f", "woman"]:
            return 1
        return 2

    @staticmethod
    def _parse_age(raw_age) -> float:
        if raw_age is None:
            return -1.0
        s = str(raw_age).strip()
        if s == "":
            return -1.0
        try:
            age = float(s)
        except ValueError:
            return -1.0
        if age <= 0 or age > 120:
            return -1.0
        return age

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, age, sex_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)
        return img, label, age, sex_id


# ---------------------------------------------------
# Model: CLIP backbone + metadata head
# ---------------------------------------------------

class ClipMetaClassifier(nn.Module):
    def __init__(self, clip_model, meta_mode: str = "image", train_backbone: bool = False):
        """
        meta_mode:
          - "image"         : 메타데이터 사용 X
          - "image_sex"     : 성별(one-hot 3차원)만 사용
          - "image_age"     : 나이(1차원, /100 스케일링)만 사용
          - "image_age_sex" : 나이 + 성별 사용
        """
        super().__init__()
        self.clip_model = clip_model
        self.meta_mode = meta_mode

        if not train_backbone:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # CLIP visual encoder output dimension
        # openai/CLIP 모델은 visual.output_dim에 임베딩 차원을 가짐
        embed_dim = self.clip_model.visual.output_dim

        meta_dim = 0
        if meta_mode in ["image_sex", "image_age_sex"]:
            meta_dim += 3  # sex one-hot (male, female, unknown)
        if meta_mode in ["image_age", "image_age_sex"]:
            meta_dim += 1  # age scalar

        self.meta_dim = meta_dim

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + meta_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, images, ages, sex_ids):
        # 이미지 임베딩
        img_feat = self.clip_model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # 메타데이터 벡터 구성
        meta_list = []

        if self.meta_mode in ["image_sex", "image_age_sex"]:
            # sex_ids: (B,) int64 -> one-hot (B, 3)
            sex_one_hot = F.one_hot(sex_ids.long(), num_classes=3).float()
            meta_list.append(sex_one_hot)

        if self.meta_mode in ["image_age", "image_age_sex"]:
            # ages: (B,) float32 -> (B, 1), 0~1 정도 스케일
            ages = ages.view(-1, 1) / 100.0
            meta_list.append(ages)

        if meta_list:
            meta_vec = torch.cat(meta_list, dim=1)
            feat = torch.cat([img_feat, meta_vec], dim=1)
        else:
            feat = img_feat

        logits = self.classifier(feat)
        return logits


# ---------------------------------------------------
# Metrics helper
# ---------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return acc, f1, prec, rec, auc


def print_summary_line(prefix: str, acc, f1, prec, rec, auc):
    print(
        f"{prefix:<22} | "
        f"Acc: {acc:.4f} | "
        f"F1: {f1:.4f} | "
        f"Precision: {prec:.4f} | "
        f"Recall: {rec:.4f} | "
        f"AUC: {auc:.4f}"
    )


# ---------------------------------------------------
# Train & Eval loops
# ---------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    all_probs = []

    running_loss = 0.0
    n_samples = 0

    for images, labels, ages, sex_ids in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        ages = torch.tensor(ages, dtype=torch.float32, device=device)
        sex_ids = torch.tensor(sex_ids, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(images, ages, sex_ids)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = running_loss / max(n_samples, 1)
    acc, f1, prec, rec, auc = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, acc, f1, prec, rec, auc


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    all_probs = []

    running_loss = 0.0
    n_samples = 0

    for images, labels, ages, sex_ids in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        ages = torch.tensor(ages, dtype=torch.float32, device=device)
        sex_ids = torch.tensor(sex_ids, dtype=torch.long, device=device)

        logits = model(images, ages, sex_ids)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = running_loss / max(n_samples, 1)
    acc, f1, prec, rec, auc = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, acc, f1, prec, rec, auc


# ---------------------------------------------------
# Argument parsing
# ---------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True,
                        help="이미지 루트 디렉토리 (예: data/processed/ham10000_bin)")
    parser.add_argument("--metadata", type=str, required=True,
                        help="메타데이터 CSV 경로 (예: ham10k_meta_for_effnet.csv)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--model-name", type=str, default="ViT-B/32",
                        help="CLIP 모델 이름 (clip.load에 들어가는 이름)")
    parser.add_argument("--meta-mode", type=str, default="image",
                        choices=["image", "image_sex", "image_age", "image_age_sex"],
                        help="메타데이터 사용 방식")
    parser.add_argument("--train-backbone", action="store_true",
                        help="지정하면 CLIP backbone도 같이 finetune")

    parser.add_argument("--image-col", type=str, default="image_id")
    parser.add_argument("--label-col", type=str, default="target")
    parser.add_argument("--sex-col", type=str, default="sex")
    parser.add_argument("--age-col", type=str, default="age")
    parser.add_argument("--split-col", type=str, default="split")

    parser.add_argument("--out-dir", type=str, default="runs_clip_meta",
                        help="모델 저장 디렉토리")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="실험 이름 (기본값: clip_<model>_<meta-mode>)")

    return parser.parse_args()


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # CLIP 모델 + 전처리 로드
    clip_model, preprocess = clip.load(args.model_name, device=device)
    print(f"Loaded CLIP model: {args.model_name}")

    # Dataset / DataLoader
    train_dataset = HamMetaCLIPDataset(
        data_root=args.data,
        metadata_csv=args.metadata,
        split="train",
        preprocess=preprocess,
        image_col=args.image_col,
        label_col=args.label_col,
        sex_col=args.sex_col,
        age_col=args.age_col,
        split_col=args.split_col,
    )

    val_dataset = HamMetaCLIPDataset(
        data_root=args.data,
        metadata_csv=args.metadata,
        split="val",
        preprocess=preprocess,
        image_col=args.image_col,
        label_col=args.label_col,
        sex_col=args.sex_col,
        age_col=args.age_col,
        split_col=args.split_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Model
    model = ClipMetaClassifier(
        clip_model=clip_model,
        meta_mode=args.meta_mode,
        train_backbone=args.train_backbone,
    ).to(device)

    # Optimizer (backbone까지 학습 여부에 따라 parameter 선택)
    if args.train_backbone:
        optim_params = model.parameters()
    else:
        # classifier만 학습 (clip backbone은 frozen)
        optim_params = model.classifier.parameters()

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Experiment name & save path
    if args.exp_name is None:
        safe_model_name = args.model_name.replace("/", "-")
        exp_name = f"clip_{safe_model_name}_{args.meta_mode}"
    else:
        exp_name = args.exp_name

    save_path = os.path.join(args.out_dir, exp_name + ".pt")
    print(f"Experiment: {exp_name}")
    print(f"Save path:  {save_path}")

    best_val_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss, tr_acc, tr_f1, tr_prec, tr_rec, tr_auc = train_one_epoch(
            model, train_loader, optimizer, device
        )
        print_summary_line("Train",
                           tr_acc, tr_f1, tr_prec, tr_rec, tr_auc)
        print(f"Train loss: {train_loss:.4f}")

        val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = eval_one_epoch(
            model, val_loader, device
        )
        print_summary_line("Val",
                           val_acc, val_f1, val_prec, val_rec, val_auc)
        print(f"Val loss:   {val_loss:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"[Save] best updated (AUC={val_auc:.4f}) -> {save_path}")

    print(f"\nBest Val AUC: {best_val_auc:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
