"""Uniform recombination with the existing truncation selection strategy.

Use EvolutionStrategy(Truncate(0.5), Recombine()) to keep population size fixed.
"""

from random import randrange

import numpy as np

from main import Generation, Individuel, MutationStrategy
from mutation import Truncate


class Recombine(MutationStrategy):
    """Produce two complementary children per parent, without bit mutation.

    Each parent chooses a random distinct mate. A population of one produces
    two independent clones. Truncate's argument is the fraction retained.
    """

    def mutate(self, generation: Generation) -> Generation:
        parents = generation.generation
        if not parents:
            return Generation([])
        if any(parent.G != parents[0].G for parent in parents):
            raise ValueError("Parents must have genomes of equal length")

        children = []
        for index, parent in enumerate(parents):
            if len(parents) == 1:
                mate = parent
            else:
                mate_index = randrange(len(parents) - 1)
                if mate_index >= index:
                    mate_index += 1
                mate = parents[mate_index]

            mask = np.random.random(parent.G) < 0.5
            children.append(Individuel(np.where(mask, parent.genome, mate.genome)))
            children.append(Individuel(np.where(mask, mate.genome, parent.genome)))

        return Generation(children)
