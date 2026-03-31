import numpy as np
from dataclasses import dataclass


@dataclass
class ImageWithMasks:
    """Модель для хранения изображения и его масок.
    Маски хранятся на диске, не в памяти, чтобы экономить RAM.
    """
    image: np.ndarray  # RGB изображение [H, W, 3]
    name: str = ""  # имя изображения

