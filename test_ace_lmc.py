#!/usr/bin/env python3
# test_ace_lmc.py — backward-compatibility stub
# Implementation moved to ace_fcn_lmc/test_ace_lmc.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ace_fcn_lmc'))
sys.path.insert(0, str(Path(__file__).parent))
from test_ace_lmc import run_evaluation, run_evaluation_lmc  # noqa: F401
if __name__ == '__main__':
    import importlib, runpy
    runpy.run_path(str(Path(__file__).parent / 'ace_fcn_lmc' / 'test_ace_lmc.py'), run_name='__main__')
