"""
configs
-------
Qubit-specific configuration definitions for QuantumRL.
"""

from .config_1qubit import Config as Config1Qubit
from .config_2qubit import Config as Config2Qubit

__all__ = ['Config1Qubit', 'Config2Qubit']
