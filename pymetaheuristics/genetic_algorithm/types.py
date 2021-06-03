from typing import Any, Callable, List, Tuple


# Variable Types
Genome = List[Any]
Population = List[Genome]

# Function Types
ConstraintFunction = Callable[[Genome], bool]
GenomeGeneratorFunction = Callable[[], Genome]
FitnessFunction = Callable[[Genome], float]
SelectionFunction = Callable[
    [Population, FitnessFunction, Any], Population]
CrossOverFunction = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunction = Callable[[Genome, Any], Genome]
