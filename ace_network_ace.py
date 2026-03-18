# ace_network_ace.py — backward-compatibility stub
# Implementation moved to ace_fcn_lmc/ace_network_ace.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ace_fcn_lmc'))
from ace_network_ace import ACEEncoder, RegressorACE  # noqa: F401
