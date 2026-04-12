#!/usr/bin/env python3
# train_ace_lmc.py — backward-compatibility stub
# Implementation moved to ace_fcn_lmc/train_ace_lmc.py
import sys
import importlib.util
from pathlib import Path

_impl = Path(__file__).parent / 'ace_fcn_lmc' / 'train_ace_lmc.py'
_spec = importlib.util.spec_from_file_location('ace_fcn_lmc.train_ace_lmc', _impl)
_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(Path(__file__).parent / 'ace_fcn_lmc'))
sys.path.insert(0, str(Path(__file__).parent))
_spec.loader.exec_module(_mod)
main = _mod.main

if __name__ == '__main__':
    main()
