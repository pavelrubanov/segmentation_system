from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class ImageWithMasks:
    """Путь к изображению и список сохранённых для него масок.

    Саму картинку не держим в памяти: на 4K-кадрах это съело бы гигабайты
    при просмотре сотен файлов. Перечитываем с диска при каждой смене.
    """
    path: str
    name: str = ""
    mask_entries: list = field(default_factory=list)  # list[MaskEntry]

    def load_image_from_disk(self) -> np.ndarray:
        return np.array(Image.open(self.path).convert("RGB"))
