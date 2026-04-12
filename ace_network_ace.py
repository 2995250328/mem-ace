# ace_network_ace.py — backward-compatibility stub
# Implementation moved to ace_fcn_lmc/ace_network_ace.py
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ace_fcn_lmc'))
sys.path.insert(0, str(Path(__file__).parent))
_impl = Path(__file__).parent / 'ace_fcn_lmc' / 'ace_network_ace.py'
_spec = importlib.util.spec_from_file_location('ace_network_ace', _impl)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['ace_network_ace'] = _mod
_spec.loader.exec_module(_mod)
ACEEncoder = _mod.ACEEncoder
RegressorACE = _mod.RegressorACE
