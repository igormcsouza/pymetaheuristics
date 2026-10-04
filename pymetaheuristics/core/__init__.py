"""Core building blocks shared by all heuristics."""
from pymetaheuristics.core.evaluation import counting
from pymetaheuristics.core.heuristic import Heuristic
from pymetaheuristics.core.problem import Direction, Problem
from pymetaheuristics.core.result import OptimizationResult
from pymetaheuristics.core.termination import (
    State, Stop, any_of, max_evaluations, max_iterations, max_time,
    target_value)

__all__ = [
    'Direction', 'Heuristic', 'OptimizationResult', 'Problem', 'State',
    'Stop', 'any_of', 'counting', 'max_evaluations', 'max_iterations',
    'max_time', 'target_value']
