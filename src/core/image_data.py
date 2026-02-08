import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class ImageWithMasks:
    """Модель для хранения изображения и его масок"""
    image: np.ndarray  # RGB изображение [H, W, 3]
    masks: List[np.ndarray] = field(default_factory=list)  # список масок [H, W]
    name: str = ""  # имя изображения

