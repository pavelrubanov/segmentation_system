"""Обучение XGBoost классификатора листьев Sphagnum на MobileNetV3 признаках.

Запуск (из segmentation_system/):
    python -m classifier.train --leaf-type capillifolium --cv 5
    python -m classifier.train --leaf-type branch
    python -m classifier.train --leaf-type girgensohnii --cv 5

Сохраняет:
    models/model_<leaf-type>.pkl          --- веса XGBoost (рядом с mobile_sam.pt)
    classifier/results/model_<type>.json  --- метрики VAL/TEST (или CV-отчёт)

Требует classifier/features.npz (собранный через `python -m classifier.mobilenet`).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier

from .variants import IDENTITY_VARIANT, VARIANTS

SEED = 42
CLASSES = ["good_leaf", "bad_leaf", "non_leaf"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
GOOD_PREFIX = "good_leaf_"
SHARED_CLASSES = ("bad_leaf", "non_leaf")

DEFAULT_DB = Path(__file__).parent / "features.npz"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).parent / "results"


# --- FeaturesDB: чтение и выборка ---


class FeaturesDB:
    """Ленивый читатель features.npz с выборкой по (file_idx, variant_idx)."""

    def __init__(self, path: Path = DEFAULT_DB):
        if not path.exists():
            raise FileNotFoundError(
                f"features.npz не найден: {path}\n"
                f"Запусти `python -m leaf_model.mobilenet --data data --out {path}`"
            )
        data = np.load(path, allow_pickle=False)
        self.X = data["X"].astype(np.float32, copy=False)
        self.file_names = list(map(str, data["file_names"]))
        self.class_dirs = list(map(str, data["class_dirs"]))
        self.n_variants = int(data["n_variants"])
        if self.n_variants != len(VARIANTS):
            raise RuntimeError(
                f"features.npz n_variants={self.n_variants} ≠ pipeline.VARIANTS={len(VARIANTS)}. Пересобери."
            )
        self._variant_idx = {v: i for i, v in enumerate(VARIANTS)}

    def list_leaf_types(self) -> list[str]:
        return sorted({
            c[len(GOOD_PREFIX):] for c in set(self.class_dirs) if c.startswith(GOOD_PREFIX)
        })

    def find_samples(self, leaf_type: str) -> list[tuple[int, int]]:
        """Возвращает [(file_idx, label), ...] для данного типа листа."""
        good = f"{GOOD_PREFIX}{leaf_type}"
        if good not in set(self.class_dirs):
            raise FileNotFoundError(
                f"{good} нет в features.npz. Доступно: {self.list_leaf_types()}"
            )
        out: list[tuple[int, int]] = []
        for i, cd in enumerate(self.class_dirs):
            if cd == good:
                out.append((i, LABEL_MAP["good_leaf"]))
            elif cd in SHARED_CLASSES:
                out.append((i, LABEL_MAP[cd]))
        return out

    def materialize(self, samples: list[tuple[int, int]],
                    variants: list = (IDENTITY_VARIANT,)
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Из списка [(file_idx, label)] и списка вариантов аугментации
        собирает (X[N*|V|, 576], y[N*|V|])."""
        v_indices = [self._variant_idx[v] for v in variants]
        rows = np.empty(len(samples) * len(v_indices), dtype=np.int64)
        labels = np.empty(len(samples) * len(v_indices), dtype=np.int64)
        k = 0
        for file_idx, label in samples:
            base = file_idx * self.n_variants
            for v in v_indices:
                rows[k] = base + v
                labels[k] = label
                k += 1
        return self.X[rows], labels

    def file_name_at(self, file_idx: int) -> str:
        return self.file_names[file_idx]


# --- метрики ---


def _cm(y_true, y_pred) -> np.ndarray:
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _per_class(cm: np.ndarray) -> list[dict]:
    out = []
    for i, cls in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out.append({"class": cls, "precision": prec, "recall": rec, "f1": f1, "support": support})
    return out


def _print_cm(cm: np.ndarray, tag: str) -> dict:
    acc = cm.diagonal().sum() / cm.sum()
    print(f"\n[{tag}] Матрица ошибок:")
    print(f"{'':>12s}" + "".join(f"{c:>12s}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>12s}" + "".join(f"{cm[i, j]:>12d}" for j in range(len(CLASSES))))
    rows = _per_class(cm)
    print(f"\n{'Класс':>12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    for r in rows:
        print(f"{r['class']:>12s} {r['precision']:>10.3f} {r['recall']:>10.3f} "
              f"{r['f1']:>10.3f} {r['support']:>10d}")
    print(f"\nOverall accuracy: {acc:.4f}")
    return {"accuracy": float(acc), "per_class": rows}


def _make_clf(args) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(CLASSES),
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        eval_metric="mlogloss",
    )


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CLASSES))
    w = len(y) / (len(CLASSES) * counts)
    return w[y]


# --- обучение ---


def _split_6_2_2(samples: list, labels: list[int]):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
    train_idx, rest = next(sss1.split(samples, labels))
    rest_labels = [labels[i] for i in rest]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_pos, test_pos = next(sss2.split(rest, rest_labels))
    return list(train_idx), [rest[p] for p in val_pos], [rest[p] for p in test_pos]


def _run_holdout(args, db: FeaturesDB, samples, labels):
    train_idx, val_idx, test_idx = _split_6_2_2(samples, labels)

    print(f"{'Класс':<12} {'Всего':>7} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("-" * 45)
    for i, cls in enumerate(CLASSES):
        cnt = lambda idx: sum(1 for j in idx if labels[j] == i)
        print(f"{cls:<12} {labels.count(i):>7} {cnt(train_idx):>7} {cnt(val_idx):>7} {cnt(test_idx):>7}")

    print(f"\nМатериализация (train x{len(VARIANTS)}, val/test x1):")
    X_train, y_train = db.materialize([samples[i] for i in train_idx], variants=VARIANTS)
    X_val,   y_val   = db.materialize([samples[i] for i in val_idx])
    X_test,  y_test  = db.materialize([samples[i] for i in test_idx])
    print(f"  train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    print(f"\nОбучение XGBoost: n_estimators={args.n_trees}, max_depth={args.max_depth}, lr={args.lr}")
    t0 = time.time()
    clf = _make_clf(args)
    clf.fit(X_train, y_train, sample_weight=_balanced_weights(y_train))
    print(f"Fit: {time.time() - t0:.1f}s")

    val_report  = _print_cm(_cm(y_val,  clf.predict(X_val)),  "VAL")
    test_report = _print_cm(_cm(y_test, clf.predict(X_test)), "TEST")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"model": clf, "classes": CLASSES, "leaf_type": args.leaf_type}, f)
    print(f"\nМодель: {args.out}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = RESULTS_DIR / f"model_{args.leaf_type}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "leaf_type": args.leaf_type,
            "n_trees": args.n_trees, "max_depth": args.max_depth, "lr": args.lr,
            "val_report": val_report, "test_report": test_report,
        }, f, indent=2, ensure_ascii=False)
    print(f"Метрики: {history_path}")


def _run_cv(args, db: FeaturesDB, samples, labels):
    k = args.cv
    label_arr = np.array(labels)
    print(f"\n[{k}-fold CV]  samples={len(samples)}  features={db.X.shape[1]}")
    print(f"  n_trees={args.n_trees}  max_depth={args.max_depth}  lr={args.lr}  variants={len(VARIANTS)}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {(label_arr == i).sum()}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_results = []
    all_errors = []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(samples)), label_arr)):
        tr = [samples[i] for i in tr_idx]
        te = [samples[i] for i in te_idx]
        print(f"\n--- Fold {fold_i + 1}/{k}  train={len(tr)}  test={len(te)} ---", flush=True)

        X_train, y_train = db.materialize(tr, variants=VARIANTS)
        X_test,  y_test  = db.materialize(te)

        clf = _make_clf(args)
        t0 = time.time()
        clf.fit(X_train, y_train, sample_weight=_balanced_weights(y_train))
        y_pred = clf.predict(X_test)
        print(f"  fit+predict: {time.time() - t0:.1f}s")

        cm = _cm(y_test, y_pred)
        acc = float(cm.diagonal().sum() / cm.sum())
        rows = _per_class(cm)
        print(f"  accuracy={acc:.4f}")
        for r in rows:
            print(f"  {r['class']:>12s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} N={r['support']}")

        fold_results.append({"fold": fold_i + 1, "accuracy": acc,
                             "per_class": rows, "cm": cm.tolist()})

        good_idx = LABEL_MAP["good_leaf"]
        for pos, (true, pred) in enumerate(zip(y_test, y_pred)):
            if true == pred:
                continue
            all_errors.append({
                "fold": fold_i + 1, "file": db.file_name_at(te[pos][0]),
                "true": CLASSES[true], "pred": CLASSES[pred],
                "error": "FP" if true == good_idx else ("FN" if pred == good_idx else "other"),
            })

    accs = [r["accuracy"] for r in fold_results]
    print(f"\n{'='*50}")
    print(f"[CV итог {k}-fold]  accuracy = {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"\n  {'Класс':>12s} {'Prec':>10s} {'Rec':>10s} {'F1':>10s}")
    for ci, cls in enumerate(CLASSES):
        precs = [r["per_class"][ci]["precision"] for r in fold_results]
        recs  = [r["per_class"][ci]["recall"]    for r in fold_results]
        f1s   = [r["per_class"][ci]["f1"]        for r in fold_results]
        print(f"  {cls:>12s}  {np.mean(precs):.3f}+/-{np.std(precs):.3f}  "
              f"{np.mean(recs):.3f}+/-{np.std(recs):.3f}  "
              f"{np.mean(f1s):.3f}+/-{np.std(f1s):.3f}")

    cm_sum = np.sum([np.array(r["cm"]) for r in fold_results], axis=0)
    print("\n  Суммарная матрица:")
    print("  " + f"{'':>12s}" + "".join(f"{c:>12s}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        print("  " + f"{cls:>12s}" + "".join(f"{cm_sum[i, j]:>12d}" for j in range(len(CLASSES))))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_path = RESULTS_DIR / f"cv{k}_{args.leaf_type}.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump({
            "k": k, "leaf_type": args.leaf_type,
            "n_trees": args.n_trees, "max_depth": args.max_depth, "lr": args.lr,
            "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
            "fold_results": fold_results, "errors": all_errors,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nCV результаты: {cv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="XGBoost классификатор на MobileNet признаках")
    p.add_argument("--leaf-type", required=True)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--out", type=Path, default=None,
                   help="Путь для весов (default: models/model_<leaf-type>.pkl)")
    p.add_argument("--n-trees", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--cv", type=int, default=0, help="K-fold CV (0 = hold-out 60/20/20)")
    args = p.parse_args()

    if args.out is None:
        args.out = MODELS_DIR / f"model_{args.leaf_type}.pkl"

    db = FeaturesDB(args.db)
    avail = db.list_leaf_types()
    if args.leaf_type not in avail:
        p.error(f"leaf-type={args.leaf_type!r} не найден. Доступные: {avail}")

    samples = db.find_samples(args.leaf_type)
    labels = [s[1] for s in samples]
    print(f"Leaf type: {args.leaf_type}  |  features: {db.X.shape[1]}  |  db: {args.db}")

    if args.cv > 0:
        _run_cv(args, db, samples, labels)
    else:
        _run_holdout(args, db, samples, labels)


if __name__ == "__main__":
    main()
