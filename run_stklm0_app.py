#!/usr/bin/env python3
"""Launch the BertPCa STKLM0 Streamlit app from the repo root."""
import os
import sys
import subprocess

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_APP_PATH  = os.path.join(_REPO_ROOT, "stklm0", "app.py")

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", _APP_PATH] + sys.argv[1:],
    cwd=_REPO_ROOT,
)
