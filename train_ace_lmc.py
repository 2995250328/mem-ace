#!/usr/bin/env python3
# train_ace_lmc.py — backward-compatibility stub
# Implementation moved to ace_fcn_lmc/train_ace_lmc.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ace_fcn_lmc'))
sys.path.insert(0, str(Path(__file__).parent))
from train_ace_lmc import main  # noqa: F401
if __name__ == '__main__':
    main()
