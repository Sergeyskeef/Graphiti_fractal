"""
Legacy Settings Module

DEPRECATED: Use core.config.get_config() instead.
This module is kept for backward compatibility.
"""

from core.config import Settings, settings, get_config

__all__ = ["Settings", "settings", "get_config"]


