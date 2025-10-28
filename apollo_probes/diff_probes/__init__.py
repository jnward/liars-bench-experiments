from __future__ import annotations

"""
Utilities for training deception-oriented difference probes.

The package exposes helpers for:
* constructing A/B appended prompts
* caching user and assistant continuation activations
* fitting logistic probes on activation differences
"""

__all__ = [
    "config",
    "cache",
    "dataset",
    "paths",
    "prompts",
    "types",
]
