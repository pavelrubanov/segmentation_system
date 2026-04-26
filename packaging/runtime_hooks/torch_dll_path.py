"""Pre-load torch's foundation DLLs before torch.__init__ runs its own loader.

Why: torch's `_load_dll_libraries` (torch/__init__.py) globs `torch/lib/*.dll`
and loads each via `LoadLibraryExW` with strict flags (no PATH search). Under
PyInstaller frozen mode this triggers WinError 1114 (DLL initialization
routine failed) on c10.dll --- its DllMain doesn't survive that load context.

Fix: pre-load c10 + its OpenMP/glue deps via plain `ctypes.CDLL` (which uses
the relaxed loader). When torch's own loop runs later, it just receives the
existing module handle instead of re-initialising the DLL.
"""
import ctypes
import os
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")
    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        for name in ("libiomp5md.dll", "torch_global_deps.dll", "c10.dll"):
            path = os.path.join(torch_lib, name)
            if os.path.isfile(path):
                ctypes.CDLL(path)
