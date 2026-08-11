"""Evaluation module: representation extraction, drift metrics, and tracking.

Re-exports the principal classes so they can be imported directly from
``superadditivity.evaluation``.
"""

from superadditivity.evaluation.representation_extractor import RepresentationExtractor
from superadditivity.evaluation.cka_analyzer import CKAAnalyzer
from superadditivity.evaluation.rsa_analyzer import RSAAnalyzer
from superadditivity.evaluation.mmd_analyzer import MMDAnalyzer
from superadditivity.evaluation.fisher_analyzer import FisherAnalyzer
from superadditivity.evaluation.centroid_analyzer import CentroidAnalyzer
from superadditivity.evaluation.drift_tracker import DriftTracker
from superadditivity.evaluation.experiment_evaluator import ExperimentEvaluator

__all__ = [
    "RepresentationExtractor",
    "CKAAnalyzer",
    "RSAAnalyzer",
    "MMDAnalyzer",
    "FisherAnalyzer",
    "CentroidAnalyzer",
    "DriftTracker",
    "ExperimentEvaluator",
]
