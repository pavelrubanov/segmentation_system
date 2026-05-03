from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

BUILD = Path(SPECPATH).resolve()                    # build/
PROJECT_ROOT = BUILD.parent                         # segmentation_system/
SRC = PROJECT_ROOT / "src"
MODELS = PROJECT_ROOT / "models"

# mobile_sam и timm активно используют динамические импорты через реестр моделей,
# PyInstaller сам их не подхватит --- собираем целиком.
sam_datas, sam_binaries, sam_hidden = collect_all("mobile_sam")
timm_datas, timm_binaries, timm_hidden = collect_all("timm")
# xgboost попадает в граф только через pickle.load(model_*.pkl) ---
# статический анализатор PyInstaller это не видит. Тащим целиком (включая xgboost.dll).
# sklearn нужен потому что XGBClassifier наследует sklearn.base.BaseEstimator.
xgb_datas, xgb_binaries, xgb_hidden = collect_all("xgboost")
skl_datas, skl_binaries, skl_hidden = collect_all("sklearn")

a = Analysis(
    [str(SRC / "main.py")],
    pathex=[str(SRC), str(PROJECT_ROOT)],
    binaries=sam_binaries + timm_binaries + xgb_binaries + skl_binaries,
    datas=sam_datas + timm_datas + xgb_datas + skl_datas + [
        (str(MODELS), "models"),
    ],
    hiddenimports=(
        sam_hidden
        + timm_hidden
        + xgb_hidden
        + skl_hidden
        + collect_submodules("classifier")
    ),
    hookspath=[],
    runtime_hooks=[str(BUILD / "runtime_hooks" / "torch_dll_path.py")],
    excludes=[
        "tkinter",
        "IPython",
        "jupyter",
        "pytest",
        "pytest_cov",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sphagnum",
    icon=None,
    console=False,
    debug=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="sphagnum",
)
