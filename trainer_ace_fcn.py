# trainer_ace_fcn.py — backward-compatibility stub
# Implementation moved to ace_fcn_lmc/trainer_ace_fcn.py
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ace_fcn_lmc'))
sys.path.insert(0, str(Path(__file__).parent))
_impl = Path(__file__).parent / 'ace_fcn_lmc' / 'trainer_ace_fcn.py'
_spec = importlib.util.spec_from_file_location('trainer_ace_fcn', _impl)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['trainer_ace_fcn'] = _mod
_spec.loader.exec_module(_mod)
TrainerACEFCN = _mod.TrainerACEFCN
TrainerACEFCNLMC = _mod.TrainerACEFCNLMC
