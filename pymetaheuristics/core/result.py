"""Algorithm-agnostic result returned by every heuristic."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of an optimization run.

    ``history`` holds one record per iteration (e.g. best value, or a
    dict of stats); its shape is up to the algorithm. ``elapsed`` is in
    seconds. ``metadata`` carries extras such as the termination reason.
    """

    best_solution: Any
    best_value: float
    history: List[Any] = field(default_factory=list)
    iterations: int = 0
    elapsed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
