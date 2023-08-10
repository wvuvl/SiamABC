"""

Modified from
Main Author of this file: FEAR
Repo: https://github.com/PinataFarms/FEARTracker/tree/main
File: https://github.com/PinataFarms/FEARTracker/blob/main/model_training/tracker

"""

from .base_tracker import Tracker
from .AEVT_tracker import FEARTracker

__all__ = ["Tracker", "FEARTracker"]
