"""Deployable OUME-Pragmatic scheduler.

The numerical transition, retrievability, grade mapping, day boundary and
calibration functions are the v1.15 reference implementation. This package
adds persistence, a session controller and deck-agnostic prompt construction
so a person can actually study with that scheduler.
"""

from oume.session import StudyEngine

__all__ = ["StudyEngine"]
__version__ = "1.16.0"
