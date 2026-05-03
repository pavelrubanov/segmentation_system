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
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier

from .variants import IDENTITY_VARIANT, VARIANTS, Variant

SEED = 42
CLASSES = ["good_leaf", "bad_leaf", "non_leaf"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
GOOD_PREFIX = "good_leaf_"
SHARED_CLASSES = ("bad_leaf", "non_leaf")

DEFAULT_DB = Path(__file__).parent / "features.npz"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).parent / "results"

Sample = tuple[int, int]  # (file_idx, label)


# --- FeaturesDB: чтение и выборка ---


class FeaturesDB:
    """Ленивый читатель features.npz с выборкой по (file_idx, variant_idx)."""

    def __init__(self, path: Path = DEFAULT_DB):
        if not path.exists():
            raise FileNotFoundError(
                f"features.npz не найден: {path}\n"
                f"Запусти `python -m classifier.mobilenet --data data --out {path}`"
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

    def find_samples(self, leaf_type: str) -> list[Sample]:
        """[(file_idx, label), ...] для данного типа листа: good_leaf_<type> + общие SHARED_CLASSES."""
        good = f"{GOOD_PREFIX}{leaf_type}"
        if good not in set(self.class_dirs):
            raise FileNotFoundError(
                f"{good} нет в features.npz. Доступно: {self.list_leaf_types()}"
            )
        out: list[Sample] = []
        for i, cd in enumerate(self.class_dirs):
            if cd == good:
                out.append((i, LABEL_MAP["good_leaf"]))
            elif cd in SHARED_CLASSES:
                out.append((i, LABEL_MAP[cd]))
        return out

    def materialize(self, samples: list[Sample],
                    variants: Sequence[Variant] = (IDENTITY_VARIANT,)
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Из (file_idx, label) и набора аугментаций собирает (X[N*|V|, FEATURE_DIM], y[N*|V|])."""
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


def _make_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion matrix + per-class precision/recall/F1/support + overall accuracy."""
    label_ids = list(range(len(CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "per_class": [
            {"class": CLASSES[i], "precision": float(prec[i]), "recall": float(rec[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i in label_ids
        ],
        "cm": cm.tolist(),
    }


def _print_report(report: dict, tag: str) -> None:
    cm = np.array(report["cm"])
    print(f"\n[{tag}] Матрица ошибок:")
    print(f"{'':>12s}" + "".join(f"{c:>12s}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>12s}" + "".join(f"{cm[i, j]:>12d}" for j in range(len(CLASSES))))
    print(f"\n{'Класс':>12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    for r in report["per_class"]:
        print(f"{r['class']:>12s} {r['precision']:>10.3f} {r['recall']:>10.3f} "
              f"{r['f1']:>10.3f} {r['support']:>10d}")
    print(f"\nOverall accuracy: {report['accuracy']:.4f}")


def _slim_report(report: dict) -> dict:
    """Для holdout JSON --- без cm (исторический формат)."""
    return {"accuracy": report["accuracy"], "per_class": report["per_class"]}


# --- обучение ---


def _make_clf(n_trees: int, max_depth: int, lr: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=lr,
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
    return (len(y) / (len(CLASSES) * counts))[y]


def _fit(clf_params: dict, X_train, y_train) -> XGBClassifier:
    clf = _make_clf(**clf_params)
    clf.fit(X_train, y_train, sample_weight=_balanced_weights(y_train))
    return clf


def _split_6_2_2(samples: list, labels: list[int]):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
    train_idx, rest = next(sss1.split(samples, labels))
    rest_labels = [labels[i] for i in rest]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_pos, test_pos = next(sss2.split(rest, rest_labels))
    return list(train_idx), [rest[p] for p in val_pos], [rest[p] for p in test_pos]


def _print_split_table(labels: list[int], splits: dict[str, list[int]]) -> None:
    print(f"{'Класс':<12} {'Всего':>7} " + " ".join(f"{name.capitalize():>7}" for name in splits))
    print("-" * (12 + 8 + 8 * len(splits)))
    for i, cls in enumerate(CLASSES):
        per_split = " ".join(
            f"{sum(1 for j in idx if labels[j] == i):>7}" for idx in splits.values()
        )
        print(f"{cls:<12} {labels.count(i):>7} {per_split}")


def _save_model(clf: XGBClassifier, out_path: Path, leaf_type: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"model": clf, "classes": CLASSES, "leaf_type": leaf_type}, f)
    print(f"\nМодель: {out_path}")


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_holdout(args, db: FeaturesDB, samples: list[Sample], labels: list[int]) -> None:
    train_idx, val_idx, test_idx = _split_6_2_2(samples, labels)
    _print_split_table(labels, {"train": train_idx, "val": val_idx, "test": test_idx})

    print(f"\nМатериализация (train x{len(VARIANTS)}, val/test x1):")
    X_train, y_train = db.materialize([samples[i] for i in train_idx], variants=VARIANTS)
    X_val,   y_val   = db.materialize([samples[i] for i in val_idx])
    X_test,  y_test  = db.materialize([samples[i] for i in test_idx])
    print(f"  train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    clf_params = {"n_trees": args.n_trees, "max_depth": args.max_depth, "lr": args.lr}
    print(f"\nОбучение XGBoost: {clf_params}")
    t0 = time.time()
    clf = _fit(clf_params, X_train, y_train)
    print(f"Fit: {time.time() - t0:.1f}s")

    val_report  = _make_report(y_val,  clf.predict(X_val));  _print_report(val_report,  "VAL")
    test_report = _make_report(y_test, clf.predict(X_test)); _print_report(test_report, "TEST")

    _save_model(clf, args.out, args.leaf_type)

    history_path = RESULTS_DIR / f"model_{args.leaf_type}.json"
    _save_json(history_path, {
        "leaf_type": args.leaf_type, **clf_params,
        "val_report":  _slim_report(val_report),
        "test_report": _slim_report(test_report),
    })
    print(f"Метрики: {history_path}")


def _run_cv(args, db: FeaturesDB, samples: list[Sample], labels: list[int]) -> None:
    k = args.cv
    label_arr = np.array(labels)
    clf_params = {"n_trees": args.n_trees, "max_depth": args.max_depth, "lr": args.lr}

    print(f"\n[{k}-fold CV]  samples={len(samples)}  features={db.X.shape[1]}")
    print(f"  {clf_params}  variants={len(VARIANTS)}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {(label_arr == i).sum()}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_results = []
    all_errors = []
    good_idx = LABEL_MAP["good_leaf"]

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(samples)), label_arr)):
        tr = [samples[i] for i in tr_idx]
        te = [samples[i] for i in te_idx]
        print(f"\n--- Fold {fold_i + 1}/{k}  train={len(tr)}  test={len(te)} ---", flush=True)

        X_train, y_train = db.materialize(tr, variants=VARIANTS)
        X_test,  y_test  = db.materialize(te)

        t0 = time.time()
        clf = _fit(clf_params, X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"  fit+predict: {time.time() - t0:.1f}s")

        report = _make_report(y_test, y_pred)
        print(f"  accuracy={report['accuracy']:.4f}")
        for r in report["per_class"]:
            print(f"  {r['class']:>12s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} N={r['support']}")
        fold_results.append({"fold": fold_i + 1, **report})

        for pos, (true, pred) in enumerate(zip(y_test, y_pred)):
            if true == pred:
                continue
            all_errors.append({
                "fold": fold_i + 1, "file": db.file_name_at(te[pos][0]),
                "true": CLASSES[true], "pred": CLASSES[pred],
                "error": "FP" if true == good_idx else ("FN" if pred == good_idx else "other"),
            })

    accs = np.array([r["accuracy"] for r in fold_results])
    print(f"\n{'='*50}")
    print(f"[CV итог {k}-fold]  accuracy = {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"\n  {'Класс':>12s} {'Prec':>10s} {'Rec':>10s} {'F1':>10s}")
    for ci, cls in enumerate(CLASSES):
        stats = {
            metric: np.array([r["per_class"][ci][metric] for r in fold_results])
            for metric in ("precision", "recall", "f1")
        }
        print(f"  {cls:>12s}  " + "  ".join(
            f"{s.mean():.3f}+/-{s.std():.3f}" for s in stats.values()
        ))

    cm_sum = np.sum([np.array(r["cm"]) for r in fold_results], axis=0)
    print("\n  Суммарная матрица:")
    print("  " + f"{'':>12s}" + "".join(f"{c:>12s}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        print("  " + f"{cls:>12s}" + "".join(f"{cm_sum[i, j]:>12d}" for j in range(len(CLASSES))))

    cv_path = RESULTS_DIR / f"cv{k}_{args.leaf_type}.json"
    _save_json(cv_path, {
        "k": k, "leaf_type": args.leaf_type, **clf_params,
        "mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std()),
        "fold_results": fold_results, "errors": all_errors,
    })
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

    (_run_cv if args.cv > 0 else _run_holdout)(args, db, samples, labels)


if __name__ == "__main__":
    main()
