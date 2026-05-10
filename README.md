<img width="1922" height="1112" alt="segmentation_canvas_autosegment_result" src="https://github.com/user-attachments/assets/dcd07a2d-d10d-44f5-8f1f-06e91d07cf5f" />

# segmentation_system

Настольное приложение для сегментации, контурирования и параметрического анализа
биологических объектов (листьев мха *Sphagnum*). MobileSAM + PyQt6, работает на CPU.

## Возможности

Приложение закрывает весь цикл работы биолога:
**снимок → выделение листа → проверка/правка → измерения → таблица для анализа**.
Главный экран предлагает три режима работы.

### 1. Сегментация изображений

Интерактивная разметка одного снимка за раз. На холсте ставите положительную
точку (ЛКМ), или отрицательную (ПКМ), или обводите bounding box (drag ЛКМ) ---
MobileSAM за ~100 мс выдаёт маску, она показывается полупрозрачным
cyan-overlay'ем. Если границу нужно поправить руками --- режим редактирования
с кистью и ластиком с регулируемым размером (1–200 px). Поддерживается зум
колесом к курсору и панорама по `Ctrl + drag`.

В этом же окне работает **авто-сегментация одного кадра** (SAM AMG) с фильтром
через классификатор: для выбранного в тулбаре типа листа на холсте остаются
только маски, размеченные как «good_leaf». Каждую найденную маску можно
открыть двойным кликом из бокового буфера и доработать кистью.

Экспорт: для всех зафиксированных масок сохраняются `*_mask_*.png` (бинарная
маска) и `*_leaf_crop_*.png` (RGB-кроп после PCA-канонизации, апекс вверху ---
готов к измерениям и обучению классификатора).

### 2. Замеры по готовым кропам

Пакетный замер для уже препроцессированных кропов (выход первого режима либо
любые внешние PNG, прошедшие тот же препроцессинг). На вход --- список файлов
`*_leaf_crop_*.png`, на выход --- по каждому кропу:

- **длина листа** = высота bbox после PCA-канонизации;
- **ширина листа** = ширина того же bbox;
- **поперечные сечения** на N равномерных долях длины (по умолчанию N=7:
  1/8, 2/8, …, 7/8) --- для каждой доли расстояние от крайнего левого до
  крайнего правого пикселя листа в этой строке;
- **PNG-визуализация** с осью длины и прорисованными сечениями (`{crop}_vis.png`);
- **строка в CSV/XLSX** с длиной, шириной и шириной в каждом сечении.

Поддерживается пересчёт пикселей в физические единицы (мм/см/…) по заданному
масштабу пикселей-на-единицу. Формат экспорта --- CSV или XLSX (в Excel в
колонку `crop` встраивается миниатюра vis-картинки, чтобы прямо в таблице
видеть качество измерений).

### 3. Авто-сегментация и замеры

Полный конвейер для пакета исходных снимков. Для каждого фото:

1. SAM AMG ищет все правдоподобные объекты, дедупликация по IoU оставляет
   уникальные маски (см. `core/leaf_pipeline.py:auto_segment` --- ускорено
   ~×10 на 4K-кадре относительно naive-реализации).
2. Каждая маска прогоняется через классификатор выбранного типа листа
   (branch / capillifolium / girgensohnii); остаются только `good_leaf`.
3. По каждому хорошему листу --- те же измерения, что и в режиме 2.

На выход --- структура папок:

```
out_dir/
├── crops/        # IMG_4321_leaf_crop_0001.png ...
├── masks/        # IMG_4321_mask_0001.png ...
├── vis/          # IMG_4321_leaf_crop_0001_vis.png ...
└── measurements.{csv,xlsx}
```

Тяжёлая работа идёт в фоне (`QThread`), есть прогресс-диалог с отменой.

### Прочее

- Все три режима работают **на CPU**: MobileSAM занимает ~80 МБ RAM,
  XGBoost-классификатор --- единицы МБ, GPU не требуется.
- Имена файлов crop/mask/vis в обоих режимах **совпадают**, поэтому результаты
  авто-режима можно скармливать обратно в режим 2 (или в обучение
  классификатора), без переименований.
- Классификатор обучается отдельно (см. [classifier/README.md](classifier/README.md));
  готовые модели для трёх типов листьев лежат в `models/`.

## Установка окружения

```bash
python -m venv src/.venv
.\src\.venv\Scripts\activate         # Windows
# source src/.venv/bin/activate      # Linux/macOS
pip install -r src/requirements.txt
```

## Запуск

```bash
cd src
python main.py
```

Аргумент `--checkpoint` позволяет указать путь к весам MobileSAM (по умолчанию
`models/mobile_sam.pt`, разрешается через `core/paths.py:resource_path`).

## Структура

```
segmentation_system/
├── src/                         исходный код приложения
│   ├── main.py                  точка входа: Predictor + PyQt6 + NavigationWindow
│   ├── core/                    бизнес-логика без UI
│   │   ├── predictor.py             обёртка над MobileSAM
│   │   ├── leaf_pipeline.py         preprocess_mask, pca_crop, auto_segment
│   │   ├── leaf_measure.py          measure_leaf, build_fractions, vis-сохранение
│   │   ├── image_data.py            ImageWithMasks
│   │   ├── io.py                    read_rgb, pluralize_ru, format_count
│   │   ├── file_naming.py           конвенция имён crop/mask/vis + Qt-фильтры
│   │   ├── sources.py               disk_source, AutoSegmentSource (для конвейера)
│   │   ├── measurement_pipeline.py  CropItem, PipelineSettings, run_pipeline
│   │   ├── export.py                save_csv, save_xlsx (с миниатюрами vis)
│   │   └── paths.py                 resource_path (dev и PyInstaller-frozen)
│   ├── ui/                      PyQt6
│   │   ├── style.py                 тема, QSS, программные иконки
│   │   ├── navigation_window.py     главный экран (3 кнопки-карточки)
│   │   ├── runner.py                SettingsDialog + блоки + PipelineWorker + runner
│   │   ├── measure_crops.py         замеры по готовым кропам
│   │   ├── measure_images.py        авто-сегментация и замеры
│   │   └── segmentation/            окно интерактивной сегментации
│   └── requirements.txt
├── classifier/                  отдельный пакет: MobileNetV3 + XGBoost (см. classifier/README.md)
├── models/                      веса MobileSAM и обученные XGBoost-классификаторы
└── tests/                       pytest
```

## Архитектура двух batch-режимов

`measure_crops` и `measure_images` --- один и тот же конвейер, отличаются только
**источником кропов**:

```
files → source(file) → CropItem'ы → measure_one → строка экспорта
                                  → vis.png в out_dir/vis/
```

- `disk_source(path) → CropItem` --- один файл = один уже препроцессированный кроп.
- `AutoSegmentSource(predictor, classifier, crops_dir, masks_dir)(path) → 0..N CropItem` ---
  внутри: `auto_segment` + classify + фильтр `good_leaf` + сохранение crop/mask PNG.

Конвейер крутится в `PipelineWorker(QThread)`. Один UI-runner (`run_with_progress`)
для обоих режимов: создаёт `QProgressDialog`, подключает сигналы, держит ссылку на
воркер до завершения (через module-level `_active_workers`).

## Конвенция имён файлов

Вся работа с именами crop/mask/vis идёт через `core/file_naming.py`. Ручная и авто
сегментация дают идентичные имена, благодаря чему результаты можно смешивать в
одной папке:

```
{stem}_leaf_crop_{idx:04d}.png   # IMG_4321_leaf_crop_0001.png
{stem}_mask_{idx:04d}.png        # IMG_4321_mask_0001.png
{crop_stem}_vis.png              # IMG_4321_leaf_crop_0001_vis.png
```

Хелперы: `crop_filename`, `mask_filename`, `vis_filename`, `crop_file_filter`,
`image_file_filter`. Не хардкодить f-строкой --- иначе ломаются Qt-фильтры
диалогов и scanner classifier'а.

## Тесты

```bash
cd src
pytest ../tests
```

## Демонстрация
