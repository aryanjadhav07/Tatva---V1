"""
config.py
---------
Central configuration entry point for QuantumRL.
Imports and re-exports whichever qubit configuration is currently active.

To switch between qubit configurations, edit the ACTIVE CONFIG import line below:
"""

import os
import sys

# Ensure quantumrl directory is on sys.path for robust config resolution
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from configs.config_1qubit import Config  # ACTIVE CONFIG — change this import line to switch qubit count
# from configs.config_2qubit import Config

__all__ = ['Config']
