"""
VERITY: Verification Engine for Reliability and Integrity Testing of LLM Yields

A lightweight, multi-signal hallucination detection framework.
"""

from .detector import HallucinationDetector
from .signals import (
    SelfConsistencySignal,
    SemanticEntropySignal,
    TokenConfidenceSignal,
    CrossExaminerSignal,
    PerplexityVarianceSignal,
    ExpertiseWeightedSignal
)
from .fusion import SignalFusion
from .utils import DetectionResult

__version__ = "0.1.0"
__all__ = [
    "HallucinationDetector",
    "SelfConsistencySignal",
    "SemanticEntropySignal",
    "TokenConfidenceSignal",
    "CrossExaminerSignal",
    "PerplexityVarianceSignal",
    "ExpertiseWeightedSignal",
    "SignalFusion",
    "DetectionResult"
]
