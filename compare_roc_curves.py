#!/usr/bin/env python3
"""
4가지 Augmentation ROC Curve 비교
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 로드
val_tf = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

val_ds = datasets.ImageFolder("data/processed/ham10000/val", transform=val_tf)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

# 모델 정보
models_info = [
    ("Baseline", "runs/ham10k_effb0_weighted/effb0_best.pt", 'blue'),
    ("Mixup (동일)", "runs/ham10k_effb0_mixup_same/effb0_best.pt", 'green'),
    ("Mixup (교차)", "runs/ham10k_effb0_mixup_cross/effb0_best.pt", 'orange'),
    ("CutMix", "runs/ham10k_effb0_cutmix/effb0_best.pt", 'red'),
]

print("\n" + "="*70)
print("ROC Curve 비교 분석")
print("="*70 + "\n")

# 결과 저장
results = []
roc_data = {}

# 각 모델 평가
for model_name, model_path, color in models_info:
    print(f"평가 중: {model_name}...")
    
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Malignant 확률
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ROC Curve 계산
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    roc_data[model_name] = {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'color': color
    }
    
    results.append({
        'Model': model_name,
        'AUC': roc_auc
    })
    
    print(f"  ✓ AUC: {roc_auc:.4f}")

print("\n" + "="*70)

# 1. ROC Curve 비교 그래프 (큰 버전)
plt.figure(figsize=(10, 8))

for model_name, data in roc_data.items():
    plt.plot(data['fpr'], data['tpr'], 
             color=data['color'], 
             lw=2.5, 
             label=f"{model_name} (AUC = {data['auc']:.4f})")

# Random Classifier
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
plt.title('ROC Curve Comparison - Data Augmentation Methods', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_comparison_large.png', dpi=300, bbox_inches='tight')
print("✓ 저장: roc_comparison_large.png")

# 2. ROC Curve 비교 그래프 (PPT용 - 더 깔끔)
plt.figure(figsize=(12, 8))

for model_name, data in roc_data.items():
    plt.plot(data['fpr'], data['tpr'], 
             color=data['color'], 
             lw=3, 
             label=f"{model_name} (AUC = {data['auc']:.3f})",
             marker='o',
             markersize=4,
             markevery=20)

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5, label='Random')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('Data Augmentation Methods - ROC Curve Comparison', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12, framealpha=0.9)
plt.grid(alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig('roc_comparison_ppt.png', dpi=300, bbox_inches='tight')
print("✓ 저장: roc_comparison_ppt.png (PPT용)")

# 3. AUC 비교 막대 그래프
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('AUC', ascending=False)

plt.figure(figsize=(10, 6))
colors_list = [roc_data[model]['color'] for model in df_results['Model']]
bars = plt.bar(df_results['Model'], df_results['AUC'], color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)

# 값 표시
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim([0.85, 1.0])  # AUC는 보통 0.85 이상
plt.ylabel('AUC Score', fontsize=13, fontweight='bold')
plt.title('AUC Score Comparison - Data Augmentation Methods', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('auc_comparison_bar.png', dpi=300, bbox_inches='tight')
print("✓ 저장: auc_comparison_bar.png")

# 4. 비교표 (CSV)
df_results.to_csv('augmentation_auc_comparison.csv', index=False)
print("✓ 저장: augmentation_auc_comparison.csv")

# 5. 결과 출력
print("\n" + "="*70)
print("AUC Score 순위")
print("="*70)
for idx, row in df_results.iterrows():
    rank = list(df_results.index).index(idx) + 1
    print(f"{rank}. {row['Model']:20s} - AUC: {row['AUC']:.4f}")
print("="*70 + "\n")

# 6. 상세 비교표 (텍스트)
print("\n" + "="*70)
print("AUC 상세 비교")
print("="*70)
print(f"{'Model':<20} {'AUC':>8} {'Baseline 대비':>15}")
print("-"*70)
baseline_auc = results[0]['AUC']
for result in results:
    diff = result['AUC'] - baseline_auc
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    print(f"{result['Model']:<20} {result['AUC']:>8.4f} {diff_str:>15}")
print("="*70)

print("\n모든 비교 그래프가 생성되었습니다!")
print("\n생성된 파일:")
print("  1. roc_comparison_large.png - 상세 버전")
print("  2. roc_comparison_ppt.png - PPT용 (추천!)")
print("  3. auc_comparison_bar.png - 막대 그래프")
print("  4. augmentation_auc_comparison.csv - 데이터")
