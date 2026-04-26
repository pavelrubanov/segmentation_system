# segmentation_system

Настольное приложение для сегментации, контурирования и параметрического анализа
биологических объектов (листьев мха *Sphagnum*). MobileSAM + PyQt6, работает на CPU.

## Сборка

```bash
python -m venv .venv
.\.venv\Scripts\activate       # Windows
# source .venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

## Запуск

```bash
cd src
python main.py
```

На главном экране — три режима:

- **Сегментация изображений** — интерактивная работа с MobileSAM
  (точки-промпты, bounding box, кисть/ластик) + авто-сегментация всех объектов.
- **Обработка масок** — пакетные измерения по готовым PNG-маскам → CSV/XLSX.
- **Автоматическая обработка** — изображения → авто-сегментация → отбор хороших листьев
  классификатором → метрики → Excel с превью каждого листа.

## Структура

```
segmentation_system/
├── src/                    исходный код приложения
│   ├── main.py            точка входа
│   ├── core/              Predictor, ImageWithMasks, leaf_measure
│   └── ui/                PyQt6: окна, канвас, диалоги
├── leaf_model/             обучаемые модели (см. leaf_model/README.md)
│   ├── classifier/        3-классовая классификация листьев
│   └── filter/            one-class anomaly detection
├── models/                 все веса: MobileSAM, классификатор, фильтры
└── tests/                  pytest
```

## Демонстрация
