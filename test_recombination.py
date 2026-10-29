import unittest
from unittest.mock import patch

import numpy as np

from main import EvolutionStrategy, Experiment, Generation, Individuel, MutationStrategy
from recombination import Recombine, Truncate


class RecombinationTests(unittest.TestCase):
    def test_children_mix_parental_alleles_without_changing_parents(self):
        parents = Generation([
            Individuel(np.array([0, 0, 1, 1])),
            Individuel(np.array([1, 1, 0, 1])),
        ])
        original = [parent.genome.copy() for parent in parents.generation]
        with patch('recombination.np.random.random', return_value=np.array([0.1, 0.9, 0.1, 0.9])):
            children = Recombine().mutate(parents)

        self.assertEqual(children.N, 4)
        np.testing.assert_array_equal(children.generation[0].genome, [0, 1, 1, 1])
        np.testing.assert_array_equal(children.generation[1].genome, [1, 0, 0, 1])
        for index in (0, 2):
            first, second = children.generation[index:index + 2]
            np.testing.assert_array_equal(first.genome + second.genome, original[0] + original[1])

        children.generation[0].genome[:] = 0
        np.testing.assert_array_equal(children.generation[1].genome, [1, 0, 0, 1])
        for parent, genome in zip(parents.generation, original):
            np.testing.assert_array_equal(parent.genome, genome)

    def test_empty_and_single_parent_populations(self):
        strategy = Recombine()
        self.assertIsInstance(strategy, MutationStrategy)
        self.assertEqual(strategy.mutate(Generation([])).N, 0)
        parent = Individuel(np.array([1, 0]))
        children = strategy.mutate(Generation([parent]))
        self.assertEqual(children.N, 2)
        for child in children.generation:
            np.testing.assert_array_equal(child.genome, parent.genome)
            self.assertFalse(np.shares_memory(child.genome, parent.genome))
        self.assertFalse(np.shares_memory(children.generation[0].genome, children.generation[1].genome))

    def test_mismatched_genome_lengths_are_rejected(self):
        parents = Generation([Individuel(np.array([0])), Individuel(np.array([0, 1]))])
        with self.assertRaisesRegex(ValueError, 'equal length'):
            Recombine().mutate(parents)

    def test_experiment_preserves_even_and_odd_population_sizes(self):
        for size in (1, 3, 4):
            with self.subTest(size=size):
                population = Generation([
                    Individuel(np.array([index % 2, 1, 0])) for index in range(size)
                ])
                experiment = Experiment(EvolutionStrategy(Truncate(0.5), Recombine()), population, 5)
                experiment.run_experiment()
                self.assertEqual(len(experiment.lineage_record), 6)
                for generation in experiment.lineage_record:
                    self.assertEqual(generation.N, size)
                    for individual in generation.generation:
                        self.assertEqual(individual.G, 3)
                        np.testing.assert_array_equal(individual.genome[1:], [1, 0])


if __name__ == '__main__':
    unittest.main()
