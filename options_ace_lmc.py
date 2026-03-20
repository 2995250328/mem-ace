# options_ace_lmc.py — backward-compatibility stub
# Implementation moved to ace_fcn_lmc/options_ace_lmc.py
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ace_fcn_lmc'))
sys.path.insert(0, str(Path(__file__).parent))
_impl = Path(__file__).parent / 'ace_fcn_lmc' / 'options_ace_lmc.py'
_spec = importlib.util.spec_from_file_location('options_ace_lmc', _impl)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['options_ace_lmc'] = _mod
_spec.loader.exec_module(_mod)
get_lmc_train_parser = _mod.get_lmc_train_parser
