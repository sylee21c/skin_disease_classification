# train_seraph.py  — Seraph 규정 반영(PyTorch + timm EfficientNet)
import os, random
from pathlib import Path
from typing import Dict, Any, List

import requests, cv2, timm, torch, pandas as pd, numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC
from tqdm import tqdm

API_BASE = "https://api.isic-archive.com/api/v2"

def assert_not_master():
    host = os.uname().nodename.lower()
    if "master" in host or "moana-master" in host:
        raise SystemExit("[SERAPH] 마스터 노드에서 실행 금지. srun/sbatch로 컴퓨트 노드에서 실행하세요.")

def get_local_data_root():
    return Path(os.getenv("SERAPH_LOCAL_DATA", "/local_datasets"))

def get_out_root():
    user = os.getenv("USER", "user")
    return Path(os.getenv("SERAPH_OUT_ROOT", f"/data/{user}/seraph_runs/isic212"))

def fetch_collection_image_ids(collection_id: int, limit: int = 200) -> List[str]:
    ids, offset = [], 0
    while True:
        url = f"{API_BASE}/collections/{collection_id}/images?limit={limit}&offset={offset}"
        r = requests.get(url, timeout=60); r.raise_for_status()
        data = r.json()
        items = data.get("results") or data
        if not items: break
        for it in items:
            if isinstance(it, dict) and "isic_id" in it:
                ids.append(it["isic_id"])
            elif isinstance(it, dict) and "image" in it and "isic_id" in it["image"]:
                ids.append(it["image"]["isic_id"])
        if len(items) < limit: break
        offset += limit
    return sorted(set(ids))

def fetch_image_meta(isic_id: str) -> Dict[str, Any]:
    url = f"{API_BASE}/images/{isic_id}"
    r = requests.get(url, timeout=60); r.raise_for_status()
    return r.json()

def download_image(isic_id: str, save_dir: Path) -> Path | None:
    url = f"{API_BASE}/images/{isic_id}/download"
    r = requests.get(url, timeout=120)
    if r.status_code != 200: return None
    ct = r.headers.get("content-type","")
    ext = ".jpg" if ("jpeg" in ct or "jpg" in ct) else (".png" if "png" in ct else ".jpg")
    out = save_dir / f"{isic_id}{ext}"
    out.write_bytes(r.content)
    return out

def build_label_from_meta(md: Dict[str, Any], scheme: str):
    norm = lambda x: str(x).strip().lower() if x is not None else ""
    d1 = norm(md.get("diagnosis_1"))
    d3 = norm(md.get("diagnosis_3"))
    if scheme == "binary_malignancy": return 1 if d1 == "malignant" else 0
    if scheme == "melanoma_vs_others": return 1 if "melanoma" in d3 else 0
    if scheme == "diagnosis_3": return md.get("diagnosis_3") or "Unknown"
    raise ValueError(scheme)

def ensure_data_ready(local_root: Path, collection_id=212, max_images=0, label_scheme="binary_malignancy", seed=42):
    random.seed(seed); np.random.seed(seed)
    data_dir = local_root / f"isic_{collection_id}"
    img_dir  = data_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = data_dir / "meta.csv"

    if meta_csv.exists():
        print(f"[info] use existing: {meta_csv}")
        return meta_csv

    print("[step] listing collection images…")
    ids = fetch_collection_image_ids(collection_id)
    if max_images > 0: ids = ids[:max_images]
    rows = []
    for isic_id in tqdm(ids, desc="fetch+download"):
        try:
            meta = fetch_image_meta(isic_id)
            md = {}
            for k in ["metadata","clinical","meta","labels"]:
                if k in meta and isinstance(meta[k], dict):
                    md.update(meta[k])
            for k in ["age_approx","anatom_site_general","sex","diagnosis_1","diagnosis_2","diagnosis_3",
                      "diagnosis_confirm_type","melanocytic","image_type"]:
                if k in meta and k not in md:
                    md[k] = meta[k]
            label = build_label_from_meta(md, label_scheme)
            saved = None
            for ext in (".jpg",".png",".jpeg"):
                p = img_dir / f"{isic_id}{ext}"
                if p.exists(): saved = p; break
            if saved is None: saved = download_image(isic_id, img_dir)
            if saved is None: continue
            rows.append({
                "isic_id": isic_id, "filepath": str(saved), "label": label,
                "diagnosis_1": md.get("diagnosis_1"), "diagnosis_2": md.get("diagnosis_2"),
                "diagnosis_3": md.get("diagnosis_3"), "diagnosis_confirm_type": md.get("diagnosis_confirm_type"),
                "melanocytic": md.get("melanocytic"), "age_approx": md.get("age_approx"),
                "sex": md.get("sex"), "anatom_site_general": md.get("anatom_site_general"),
                "image_type": md.get("image_type"),
            })
        except Exception as e:
            print(f"[warn] {isic_id}: {e}")
            continue
    if not rows: raise RuntimeError("no images prepared")
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    return meta_csv

class LesionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scheme: str, tf=None, class_to_idx=None):
        self.df = df.reset_index(drop=True); self.scheme = scheme; self.tf = tf
        self.class_to_idx = class_to_idx or {}
        if scheme == "diagnosis_3" and not class_to_idx:
            classes = sorted(self.df["label"].astype(str).unique())
            self.class_to_idx = {c:i for i,c in enumerate(classes)}
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r.filepath, cv2.IMREAD_COLOR)
        if img is None: img = np.zeros((512,512,3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.tf: img = self.tf(image=img)["image"]
        y = self.class_to_idx[str(r.label)] if self.scheme=="diagnosis_3" else int(r.label)
        return img, y

def get_transforms(sz=384):
    train = A.Compose([
        A.LongestMaxSize(max_size=sz),
        A.PadIfNeeded(sz, sz, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomResizedCrop(sz, sz, scale=(0.8,1.0), p=0.9),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(0.05, 0.1, 15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.Normalize(), ToTensorV2(),
    ])
    val = A.Compose([
        A.LongestMaxSize(max_size=sz),
        A.PadIfNeeded(sz, sz, border_mode=cv2.BORDER_REFLECT_101),
        A.CenterCrop(sz, sz), A.Normalize(), ToTensorV2(),
    ])
    return train, val

def build_model(arch, num_classes):
    return timm.create_model(arch, pretrained=True, num_classes=num_classes)

def compute_metrics(logits, targets, scheme, num_classes):
    probs = torch.softmax(logits, 1)
    if scheme in ["binary_malignancy","melanoma_vs_others"]:
        auroc = BinaryAUROC().to(logits.device)(probs[:,1], targets)
    else:
        auroc = MulticlassAUROC(num_classes=num_classes, average="macro").to(logits.device)(probs, targets)
    return {"auroc": float(auroc)}

@torch.no_grad()
def validate(model, dl, device, crit, scheme, num_classes):
    model.eval()
    total, n = 0.0, 0
    all_logits, all_tgts = [], []
    for x,y in tqdm(dl, desc="valid", leave=False):
        x,y = x.to(device), y.to(device)
        logits = model(x); loss = crit(logits, y)
        total += loss.item(); n += 1
        all_logits.append(logits); all_tgts.append(y)
    logits = torch.cat(all_logits); tgts = torch.cat(all_tgts)
    m = compute_metrics(logits, tgts, scheme, num_classes)
    return total/max(1,n), m

def train_one_epoch(model, dl, opt, scaler, device, crit, accum=1):
    model.train()
    total, n = 0.0, 0
    for x,y in tqdm(dl, desc="train", leave=False):
        x,y = x.to(device), y.to(device)
        with autocast():
            logits = model(x); loss = crit(logits, y)/accum
        scaler.scale(loss).backward()
        if (n+1) % accum == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        total += loss.item()*accum; n += 1
    return total/max(1,n)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection_id", type=int, default=212)
    ap.add_argument("--label_scheme", type=str, default="binary_malignancy",
                    choices=["binary_malignancy","melanoma_vs_others","diagnosis_3"])
    ap.add_argument("--arch", type=str, default="tf_efficientnet_b0_ns")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_accum", type=int, default=1)
    args = ap.parse_args()

    assert_not_master()
    local_root = get_local_data_root()    # /local_datasets
    out_root   = get_out_root()           # /data/$USER/seraph_runs/isic212
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    meta_csv = ensure_data_ready(local_root, args.collection_id, args.max_images, args.label_scheme, args.seed)
    df = pd.read_csv(meta_csv)

    # classes
    class_to_idx, num_classes = None, 2
    if args.label_scheme == "diagnosis_3":
        classes = sorted(df["label"].astype(str).unique())
        class_to_idx = {c:i for i,c in enumerate(classes)}
        num_classes = len(classes)

    tr_idx, va_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df["label"].astype(str)
    )
    df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    tf_tr, tf_va = get_transforms(args.img_size)
    ds_tr = LesionDataset(df_tr, args.label_scheme, tf_tr, class_to_idx)
    ds_va = LesionDataset(df_va, args.label_scheme, tf_va, class_to_idx)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size*2, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    best, best_path = -1.0, out_root / "checkpoints" / f"{args.arch}_{args.label_scheme}_best.pth"
    for e in range(1, args.epochs+1):
        print(f"\n[Epoch {e}/{args.epochs}]")
        tr_loss = train_one_epoch(model, dl_tr, opt, scaler, device, crit, accum=args.grad_accum)
        va_loss, m = validate(model, dl_va, device, crit, args.label_scheme, num_classes)
        print(f"train {tr_loss:.4f} | valid {va_loss:.4f} | AUROC {m['auroc']:.4f}")
        if m["auroc"] > best:
            best = m["auroc"]
            torch.save({
                "model": model.state_dict(),
                "arch": args.arch,
                "label_scheme": args.label_scheme,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx
            }, best_path)
            print(f"[save] {best_path} (AUROC={best:.4f})")
    print(f"[DONE] best AUROC={best:.4f} | ckpt={best_path}")

if __name__ == "__main__":
    main()
