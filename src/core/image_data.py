import numpy as np
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class ImageWithMasks:
    """Модель для хранения изображения и его масок.
    Маски хранятся на диске, не в памяти, чтобы экономить RAM.
    Изображение загружается лениво при первом обращении.
    """
    path: str              # путь к файлу
    name: str = ""         # имя изображения
    _image: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            self._image = np.array(Image.open(self.path).convert("RGB"))
        return self._image

