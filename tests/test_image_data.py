"""Тесты модуля core/image_data.py."""
import numpy as np

from core.image_data import ImageWithMasks


class TestImageWithMasks:

    def test_default_name_empty(self):
        """По умолчанию name = ''."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        data = ImageWithMasks(image=img)
        assert data.name == ""

    def test_custom_name(self):
        """Имя сохраняется корректно."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        data = ImageWithMasks(image=img, name="leaf.jpg")
        assert data.name == "leaf.jpg"

    def test_image_stored_by_reference(self):
        """Массив изображения хранится как есть."""
        img = np.random.randint(0, 256, (50, 80, 3), dtype=np.uint8)
        data = ImageWithMasks(image=img)
        np.testing.assert_array_equal(data.image, img)

    def test_image_shape_preserved(self):
        """Форма массива не изменяется."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        data = ImageWithMasks(image=img)
        assert data.image.shape == (200, 300, 3)

    def test_image_dtype_preserved(self):
        """Тип данных массива не изменяется."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        data = ImageWithMasks(image=img)
        assert data.image.dtype == np.uint8
