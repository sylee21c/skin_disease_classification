import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--img", type=int, default=288)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ 데이터셋 로드
test_tf = transforms.Compose([
    transforms.Resize((args.img, args.img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

test_ds = datasets.ImageFolder(os.path.join(args.data, "test"), transform=test_tf)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
class_names = test_ds.classes

# 2️⃣ 모델 로드
num_classes = len(class_names)
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device)
model.eval()

# 3️⃣ 예측
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 4️⃣ 리포트 출력
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 5️⃣ 혼동 행렬 시각화
cm = confusion_matrix(y_true, y_pred)
cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7,6))
sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
print("✅ Confusion matrix saved as confusion_matrix.png")
