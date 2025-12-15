import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_DIR = "/data/blackponge/skinproj/data/raw/ham10000"
META_PATH = "/data/blackponge/skinproj/data/ham10000_metadata.csv"
OUT_DIR = "/data/blackponge/skinproj/data/processed/ham10000"

SPLIT_RATIO = [0.7, 0.15, 0.15]  # train, val, test

os.makedirs(OUT_DIR, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

meta = pd.read_csv(META_PATH)

# diagnosis_1을 대표 라벨로 사용
if "diagnosis_1" in meta.columns:
    meta = meta.rename(columns={"diagnosis_1": "diagnosis"})
else:
    raise ValueError("메타데이터에 diagnosis_1 열이 없습니다. CSV 헤더를 확인하세요.")

meta = meta[["isic_id", "diagnosis"]]
print(f"총 {len(meta)}개 이미지 메타데이터 로드 완료")

train_df, temp_df = train_test_split(
    meta, test_size=(1 - SPLIT_RATIO[0]),
    stratify=meta["diagnosis"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=(SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2])),
    stratify=temp_df["diagnosis"], random_state=42
)

splits = {"train": train_df, "val": val_df, "test": test_df}

for split, df in splits.items():
    print(f"\n{split.upper()} 세트 ({len(df)}개)")
    for cls in df["diagnosis"].unique():
        os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row["isic_id"]
        cls = row["diagnosis"]
        src = os.path.join(RAW_DIR, f"{img_id}.jpg")
        dst = os.path.join(OUT_DIR, split, cls, f"{img_id}.jpg")
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"{src} not found!")

print("데이터 정리 완료.")
