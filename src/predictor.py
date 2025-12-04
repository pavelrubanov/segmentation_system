import torch
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor

class Predictor:
    def __init__(self, checkpoint_path: str):
        model_type = "vit_t"
        mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        mobile_sam.to(device="cpu")
        mobile_sam.eval()

        self.sam_predictor = SamPredictor(mobile_sam)

    def set_image(self, image):
        self.sam_predictor.set_image(image)

    def predict(self, pos_points, neg_points, box):
        has_points = (len(pos_points) + len(neg_points)) > 0
        has_box = box is not None

        if not has_points and not has_box:
            return None

        point_coords = None
        point_labels = None
        if (len(pos_points) + len(neg_points)) > 0:
            pts_all = pos_points + neg_points
            point_coords = np.array(pts_all, dtype=np.float32)
            point_labels = np.array(
                [1] * len(pos_points) + [0] * len(neg_points),
                dtype=np.int32,
            )

        box_np = None
        if box is not None:
            box_np = np.array(box, dtype=np.float32)  # XYXY

        with torch.no_grad():
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_np,
                multimask_output=True,
            )

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]

        return best_mask