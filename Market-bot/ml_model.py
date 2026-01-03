# Compatibility shim: re-export ML model for legacy imports
# This allows `from ml_model import MLDecisionModel` to work
from ml.inference import MLDecisionModel

__all__ = ["MLDecisionModel"]
