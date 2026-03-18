# options_ace_lmc.py — backward-compatibility stub
# Implementation moved to ace_fcn_lmc/options_ace_lmc.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ace_fcn_lmc'))
from options_ace_lmc import get_lmc_train_parser  # noqa: F401
