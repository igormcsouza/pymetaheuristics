from typing import Any, Callable, List


# Variable Types
Genome = List[Any]
Population = List[Genome]

# Function Types
ConstraintFunction = Callable[[Genome], bool]
GenomeGeneratorFunction = Callable[[], Genome]
FitnessFunction = Callable[[Genome], float]
