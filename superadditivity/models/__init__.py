"""Model architectures and utilities for the superadditivity project.

Re-exports
----------
CIFARResNet, build_resnet18_cifar, build_resnet34_cifar
    CIFAR-adapted ResNet variants.
SimpleConvNet, build_convnet
    Lightweight 4-layer ConvNet for FL benchmarks.
WideResNet, build_wide_resnet
    WideResNet (WRN-28-k) for width-sensitivity analysis.
VisionTransformer, build_vit_tiny
    Vision Transformer for architecture-class generalization.
init_weights, clone_model, count_parameters, average_state_dicts, consensus_error
    Model utility functions.
"""

from superadditivity.models.convnet import SimpleConvNet, build_convnet
from superadditivity.models.model_utils import (
    average_state_dicts,
    clone_model,
    consensus_error,
    count_parameters,
    init_weights,
)
from superadditivity.models.resnet import (
    CIFARResNet,
    build_resnet18_cifar,
    build_resnet34_cifar,
)
from superadditivity.models.vit import VisionTransformer, build_vit_tiny
from superadditivity.models.wide_resnet import WideResNet, build_wide_resnet

__all__ = [
    "CIFARResNet",
    "build_resnet18_cifar",
    "build_resnet34_cifar",
    "SimpleConvNet",
    "build_convnet",
    "WideResNet",
    "build_wide_resnet",
    "VisionTransformer",
    "build_vit_tiny",
    "init_weights",
    "clone_model",
    "count_parameters",
    "average_state_dicts",
    "consensus_error",
]
