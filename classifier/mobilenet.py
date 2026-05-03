"""Извлечение признаков MobileNetV3-Small для листьев.

На вход подаётся уже препроцессированный RGB-кроп (`*_leaf_crop_*.png`,
сохранённый приложением через `core.leaf_pipeline.transform_leaf`).

Внутри: letterbox до 512×512, ImageNet-нормализация, прогон через
MobileNetV3-Small (frozen), на выходе --- 576-мерный вектор.

Модуль и для inference (CPU singleton, без device-параметра), и для
одноразовой сборки features.npz для обучения (build_features_db явно
поднимает CUDA через _ensure_model перед прогоном).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
from joblib import Parallel, delayed
from PIL import Image

from .variants import VARIANTS, Variant, apply_variant

INPUT_SIZE = 512
FEATURE_DIM = 576
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

INFERENCE_DEVICE = "cpu"

# Дубликат CROP_INFIX из src/core/file_naming.py. Пакет classifier намеренно
# держится независимым (свой venv, не импортирует из src/), поэтому константа
# повторена. При изменении --- править оба места.
_CROP_INFIX = "leaf_crop"

# Модель лениво грузится при первом использовании.
_model: torch.nn.Module | None = None
_device: torch.device | None = None


def _ensure_model(device: str | torch.device | None = None) -> tuple[torch.nn.Module, torch.device]:
    """Без явного device --- грузит на INFERENCE_DEVICE (CPU). Сборка features
    передаёт `device='cuda'` явно перед прогоном."""
    global _model, _device
    if _model is None:
        device = torch.device(INFERENCE_DEVICE) if device is None else torch.device(device)
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier = torch.nn.Identity()
        m = m.eval().to(device)
        for p in m.parameters():
            p.requires_grad = False
        _model, _device = m, device
    return _model, _device


def _fit_to_square(crop: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Letterbox в size×size с сохранением aspect ratio (чёрные поля)."""
    h, w = crop.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


@torch.no_grad()
def _forward_squares(squares: np.ndarray, batch_size: int) -> np.ndarray:
    """Letterbox-нутые [N, INPUT_SIZE, INPUT_SIZE, 3] uint8 → [N, FEATURE_DIM] float32.

    Использует загруженную через `_ensure_model` модель (на её device).
    """
    model, dev = _ensure_model()
    mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    out = np.empty((len(squares), FEATURE_DIM), dtype=np.float32)
    for s in range(0, len(squares), batch_size):
        batch = torch.from_numpy(squares[s:s + batch_size]).to(dev).float() / 255.0
        batch = (batch.permute(0, 3, 1, 2) - mean) / std
        out[s:s + len(batch)] = model(batch).cpu().numpy()
    return out


def encode(crops: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    """Из списка препроцессированных RGB-кропов получить матрицу признаков [N, FEATURE_DIM]."""
    if not crops:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    squares = np.stack([_fit_to_square(c) for c in crops])
    return _forward_squares(squares, batch_size)


# --- одноразовая сборка features.npz из data/ ---


def _transform_raw_file(crop_path: str, variants: tuple[Variant, ...]) -> np.ndarray:
    """Прочитать *_leaf_crop_*.png и сгенерировать [|variants|, INPUT_SIZE, INPUT_SIZE, 3]
    uint8. Crop --- уже результат `transform_leaf`, тут только аугментация и letterbox."""
    crop = np.array(Image.open(crop_path).convert("RGB"))
    return np.stack([_fit_to_square(apply_variant(crop, v)) for v in variants])


def _scan_class(cls_dir: Path) -> list[Path]:
    if not cls_dir.is_dir():
        return []
    return sorted(p for p in cls_dir.iterdir()
                  if f"_{_CROP_INFIX}_" in p.name and p.suffix == ".png")


def build_features_db(
    data_dir: Path,
    out_path: Path,
    classes: list[str] | None = None,
    batch_size: int = 32,
    n_jobs: int = -1,
) -> None:
    """data/<class>/*_leaf_crop_*.png в features.npz (N_files × N_variants строк × FEATURE_DIM).

    Структура npz:
        X            float32 [N*|variants|, FEATURE_DIM]
        file_names   str     [N]
        class_dirs   str     [N]
        n_variants   int32   (= 6)
    """
    if classes is None:
        classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())

    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    model, device = _ensure_model(train_device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV3-Small: {n_params:,} параметров (frozen)")
    print(f"Device: {device}  |  input: {INPUT_SIZE}x{INPUT_SIZE}  |  batch: {batch_size}")
    print(f"Классы: {classes}\n")

    all_X: list[np.ndarray] = []
    all_file_names: list[str] = []
    all_class_dirs: list[str] = []
    variants = tuple(VARIANTS)
    t0 = time.time()

    for cls_name in classes:
        files = _scan_class(data_dir / cls_name)
        if not files:
            print(f"  [skip] {cls_name}: 0 файлов")
            continue

        n = len(files)
        print(f"  [{cls_name}] {n} x {len(VARIANTS)} = {n * len(VARIANTS)} сэмплов")
        ts = time.time()
        squares_per_file = Parallel(n_jobs=n_jobs, verbose=5, prefer="processes")(
            delayed(_transform_raw_file)(str(f), variants) for f in files
        )
        squares = np.concatenate(squares_per_file, axis=0)  # [n*6, INPUT_SIZE, INPUT_SIZE, 3]
        t_pre = time.time() - ts

        ts2 = time.time()
        feats = _forward_squares(squares, batch_size)
        t_inf = time.time() - ts2

        all_X.append(feats)
        all_file_names.extend(f.name for f in files)
        all_class_dirs.extend([cls_name] * len(files))
        print(f"    preprocess {t_pre:.1f}s | inference {t_inf:.1f}s | +{feats.shape}")

    X = np.concatenate(all_X, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        file_names=np.array(all_file_names),
        class_dirs=np.array(all_class_dirs),
        n_variants=np.int32(len(VARIANTS)),
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nГотово за {time.time() - t0:.1f}s")
    print(f"  -> {out_path}  ({size_mb:.1f} MB)")
    print(f"  X = {X.shape}  files = {len(all_file_names)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Сборка features.npz (MobileNetV3-Small)")
    p.add_argument("--data", type=Path, required=True, help="Папка data/<class>/*.png")
    p.add_argument("--out",  type=Path, required=True, help="Путь к features.npz")
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()
    build_features_db(args.data, args.out, args.classes, args.batch_size, args.n_jobs)


if __name__ == "__main__":
    main()
