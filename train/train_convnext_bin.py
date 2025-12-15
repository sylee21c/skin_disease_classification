#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score
)
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to data directory")
parser.add_argument("--model", type=str, required=True, help="학습된 모델 .pt 경로")
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--img", type=int, default=288)
parser.add_argument(
    "--backbone",
    type=str,
    default="convnext_tiny",
    help="timm 모델 이름 (예: convnext_tiny, convnext_small, convnextv2_tiny 등)"
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 로드
val_tf = transforms.Compose([
    transforms.Resize((args.img, args.img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

val_path = os.path.join(args.data, "val")
val_ds = datasets.ImageFolder(val_path, transform=val_tf)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

class_names = val_ds.classes
num_classes = len(class_names)
print(f"클래스: {class_names}")
print(f"검증 데이터: {len(val_ds)} 샘플\n")

# 모델 로드 (ConvNeXt 기본)
print(f"백본(backbone): {args.backbone}, num_classes = {num_classes}")
model = timm.create_model(
    args.backbone,
    pretrained=False,
    num_classes=num_classes
).to(device)

state = torch.load(args.model, map_location=device)
model.load_state_dict(state)
model.eval()

# 예측 수행
all_labels = []
all_preds = []
all_probs = []  # 확률값 저장 (ROC curve용)

print("예측 진행 중...")
with torch.no_grad():
    for imgs, labels in val_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)  # 확률로 변환
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# 결과 저장 디렉토리
output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 1. 기본 성능 지표
# =========================
print("\n" + "="*60)
print("기본 성능 지표 (Threshold = 0.5)")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# =========================
# 2. Confusion Matrix
# =========================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Threshold = 0.5)')
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix_th0.5.png", dpi=300)
print(f"\n✓ Confusion Matrix 저장: {output_dir}/confusion_matrix_th0.5.png")

# =========================
# 3. ROC Curve & AUC (Binary Classification)
# =========================
if num_classes == 2:
    # Malignant(악성)를 positive class로 가정
    malignant_idx = class_names.index('malignant') if 'malignant' in class_names else 1
    print(f"Malignant 클래스 인덱스: {malignant_idx}")
    
    # Malignant일 확률 추출
    malignant_probs = all_probs[:, malignant_idx]
    
    # ROC curve 계산
    fpr, tpr, thresholds = roc_curve(all_labels, malignant_probs, pos_label=malignant_idx)
    roc_auc = auc(fpr, tpr)
    
    # ROC curve 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity/Recall)')
    plt.title('ROC Curve - Malignant Detection')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=300)
    print(f"✓ ROC Curve 저장: {output_dir}/roc_curve.png")
    print(f"  AUC Score: {roc_auc:.4f}")

# =========================
# 4. Threshold 최적화 실험
# =========================
print("\n" + "="*60)
print("Threshold 최적화 실험")
print("="*60)

if num_classes == 2:
    test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    
    for th in test_thresholds:
        # 새로운 threshold로 예측
        preds_at_th = (malignant_probs >= th).astype(int)
        
        acc = (preds_at_th == all_labels).mean()
        f1 = f1_score(all_labels, preds_at_th)
        precision = precision_score(all_labels, preds_at_th, zero_division=0)
        recall = recall_score(all_labels, preds_at_th, zero_division=0)
        
        results.append({
            'threshold': th,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        print(f"\nThreshold = {th:.1f}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        # Confusion matrix for each threshold
        cm_th = confusion_matrix(all_labels, preds_at_th)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_th, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Threshold = {th})')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_th{th}.png", dpi=300)
    
    print(f"\n✓ 각 Threshold별 Confusion Matrix 저장됨")
    
    # Threshold별 성능 비교 그래프
    import pandas as pd
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    titles = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(df_results['threshold'], df_results[metric], marker='o', linewidth=2)
        ax.set_xlabel('Threshold')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs Threshold')
        ax.grid(alpha=0.3)
        ax.set_xticks(test_thresholds)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/threshold_comparison.png", dpi=300)
    print(f"✓ Threshold 비교 그래프: {output_dir}/threshold_comparison.png")
    
    # 최적 threshold 찾기 (F1 기준)
    best_idx = df_results['f1'].idxmax()
    best_th = df_results.loc[best_idx, 'threshold']
    best_f1 = df_results.loc[best_idx, 'f1']
    
    print("\n" + "="*60)
    print(f"최적 Threshold (F1 기준): {best_th}")
    print(f"  F1-Score: {best_f1:.4f}")
    print(f"  Accuracy: {df_results.loc[best_idx, 'accuracy']:.4f}")
    print(f"  Precision: {df_results.loc[best_idx, 'precision']:.4f}")
    print(f"  Recall: {df_results.loc[best_idx, 'recall']:.4f}")
    print("="*60)

# =========================
# 5. Precision-Recall Curve
# =========================
if num_classes == 2:
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
        all_labels, malignant_probs, pos_label=malignant_idx
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, lw=2)
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_curve.png", dpi=300)
    print(f"✓ Precision-Recall Curve: {output_dir}/precision_recall_curve.png")

print(f"\n\n모든 평가 결과가 '{output_dir}' 폴더에 저장되었습니다.")
