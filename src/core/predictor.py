"""Обёртка над MobileSAM. Маски возвращаются как есть (uint8 0/255), без
препроцессинга --- его делает leaf_pipeline на этапе сохранения/измерений."""
from __future__ import annotations

import warnings

import numpy as np
import torch

# MobileSAM импортирует timm по устаревшим путям и перерегистрирует tiny_vit_*
# в реестре timm. Оба warning'а безвредные --- глушим только на момент импорта.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"mobile_sam\..*")
    from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

INFERENCE_DEVICE = "cpu"


class Predictor:
    def __init__(self, checkpoint_path: str):
        self.device = INFERENCE_DEVICE
        mobile_sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        mobile_sam.to(device=self.device).eval()

        self.sam_predictor = SamPredictor(mobile_sam)
        self._mask_gen = SamAutomaticMaskGenerator(
            mobile_sam,
            points_per_side=8,
            points_per_batch=64,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.88
        )

    def set_image(self, image: np.ndarray) -> None:
        self.sam_predictor.set_image(image)

    def predict(
        self,
        pos_points: list[tuple[float, float]],
        neg_points: list[tuple[float, float]],
        box: tuple[float, float, float, float] | None,
    ) -> np.ndarray | None:
        """Лучшая из 3 SAM-масок по точкам и/или bbox.
        Размер маски совпадает с исходным изображением (uint8 0/255).
        None --- если промптов нет."""
        n_pts = len(pos_points) + len(neg_points)
        if n_pts == 0 and box is None:
            return None

        coords = labels = None
        if n_pts > 0:
            coords = np.array(pos_points + neg_points, dtype=np.float32)
            labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)

        box_np = np.array(box, dtype=np.float32) if box is not None else None

        with torch.no_grad():
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=coords,
                point_labels=labels,
                box=box_np,
                multimask_output=True,
            )

        best = masks[int(np.argmax(scores))]
        return (best * 255).astype(np.uint8)

    def predict_all(
        self,
        image: np.ndarray,
        min_area_frac: float = 0.001,
        max_area_frac: float = 0.7,
    ) -> list[np.ndarray]:
        """Автосегментация (SAM AMG). Площадь маски считается долей от
        всего кадра: меньше min_area_frac --- мусор, больше max_area_frac ---
        захват фона целиком."""
        with torch.inference_mode():
            anns = self._mask_gen.generate(image)
        total = image.shape[0] * image.shape[1]
        min_px = int(total * min_area_frac)
        max_px = int(total * max_area_frac)
        return [
            (a["segmentation"].astype(np.uint8) * 255)
            for a in anns
            if min_px <= a["area"] <= max_px
        ]
