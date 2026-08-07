"""
Config package wrapper for external configuration imports.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

__all__ = ["Config"]
