# trainer_ace_fcn.py — backward-compatibility stub
# Implementation moved to ace_fcn_lmc/trainer_ace_fcn.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ace_fcn_lmc'))
from trainer_ace_fcn import TrainerACEFCN, TrainerACEFCNLMC  # noqa: F401
