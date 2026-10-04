"""Deprecated alias of ``mutations`` (old misspelled name); remove in 0.3."""
import warnings

from pymetaheuristics.genetic_algorithm.steps.mutations import *  # noqa: F401,F403
from pymetaheuristics.genetic_algorithm.steps.mutations import inter_mutation  # noqa: F401

warnings.warn(
    "pymetaheuristics.genetic_algorithm.steps.multations is deprecated; "
    "import from steps.mutations instead (removal tracked in issue #50).",
    DeprecationWarning, stacklevel=2)
