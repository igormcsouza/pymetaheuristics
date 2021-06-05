from typing import Any, Callable, Dict, List, Tuple, Union


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

# Model variables
GeneticAlgorithmHistory = Dict[
    float, Dict[
        str, Union[
            List[Tuple[List[Any], float]],
            Tuple[List[Any], float],
            float,
            Dict[str, Union[str, int, str, bool, Dict[str, Any]]]
        ]
    ]
]
