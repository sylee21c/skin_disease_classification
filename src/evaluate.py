import torch
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import timm
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 3
    
    # 학습 때와 동일하게 timm으로 모델 뼈대를 생성
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    
    # 저장된 가중치 불러오기
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval() # 평가 모드로 설정 (Dropout, BatchNorm 고정)

    # 테스트 데이터셋을 위한 변환 (데이터 증강 X)
    test_transform = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = datasets.ImageFolder(os.path.join(args.data, "test"), transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []

    # 기울기 계산 비활성화 (메모리/속도 최적화)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = test_dataset.classes
    print(f"▶ Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}\n")
    print("▶ Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # 혼동 행렬(Confusion Matrix) 생성 및 저장
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    print("\n▶ Confusion Matrix saved to 'confusion_matrix.png'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the HAM10000 dataset.')
    parser.add_argument('--data', type=str, default='data/processed/ham10000', help='Path to dataset directory')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model .pt file')
    parser.add_argument('--img', type=int, default=288, help='Image size used for training')
    args = parser.parse_args()
    evaluate(args)
