#!/usr/bin/env python3
# test_ace_dinov2_lmc.py — backward-compatibility stub
# Implementation moved to ace_dinov2_lmc/test_ace_dinov2_lmc.py
import sys, importlib.util
from pathlib import Path

_impl = Path(__file__).parent / 'ace_dinov2_lmc' / 'test_ace_dinov2_lmc.py'
_spec = importlib.util.spec_from_file_location('ace_dinov2_lmc.test_ace_dinov2_lmc', _impl)
_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(Path(__file__).parent / 'ace_dinov2_lmc'))
sys.path.insert(0, str(Path(__file__).parent))
_spec.loader.exec_module(_mod)
run_evaluation_lmc = _mod.run_evaluation_lmc

if __name__ == '__main__':
    import runpy
    runpy.run_path(str(_impl), run_name='__main__')
