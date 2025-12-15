#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"

val_tf = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

val_ds = datasets.ImageFolder("data/processed/ham10000/val", transform=val_tf)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

models_info = [
    ("Baseline", "runs/ham10k_effb0_weighted/effb0_best.pt"),
    ("Mixup (동일)", "runs/ham10k_effb0_mixup_same/effb0_best.pt"),
    ("Mixup (교차)", "runs/ham10k_effb0_mixup_cross/effb0_best.pt"),
    ("CutMix", "runs/ham10k_effb0_cutmix/effb0_best.pt"),
]

print("\n" + "="*80)
print("전체 모델 성능 비교")
print("="*80 + "\n")

results = []

for model_name, model_path in models_info:
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Malignant 확률
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 성능 지표 계산
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    results.append({
        'model': model_name,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    })
    
    print(f"{model_name:20s} | Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f}")

print("\n" + "="*80)
print("최고 성능 모델")
print("="*80)

best_acc = max(results, key=lambda x: x['accuracy'])
best_f1 = max(results, key=lambda x: x['f1'])
best_auc = max(results, key=lambda x: x['auc'])

print(f"최고 Accuracy: {best_acc['model']} ({best_acc['accuracy']:.4f})")
print(f"최고 F1-Score: {best_f1['model']} ({best_f1['f1']:.4f})")
print(f"최고 AUC:      {best_auc['model']} ({best_auc['auc']:.4f})")
print("="*80 + "\n")

# 결과를 CSV로 저장
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('model_comparison_results.csv', index=False)
print("✓ 결과가 'model_comparison_results.csv'에 저장되었습니다.")
