import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

from main import EvolutionStrategy, Experiment, Generation, Individuel
from mutation import Mutate, Truncate, mutate_individuel


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.population = Generation([
            Individuel(np.array([0, 0])),
            Individuel(np.array([1, 0])),
            Individuel(np.array([0, 1])),
            Individuel(np.array([1, 1])),
        ])

    def test_truncate_keeps_fittest_fraction(self):
        for fraction, expected in [(0, []), (0.25, [2]),
                                   (0.5, [1, 2]), (1, [0, 1, 1, 2])]:
            with self.subTest(fraction=fraction):
                selected = Truncate(fraction).select(self.population)
                self.assertEqual([i.fitness() for i in selected.generation], expected)
        self.assertEqual(Truncate(0.5).select(Generation([])).N, 0)

    def test_mutation_samples_each_bit_without_changing_parent(self):
        parent = self.population.generation[1]
        with patch('mutation.random', side_effect=[0.1, 0.9]) as random:
            result = mutate_individuel(parent, 0.5)
        np.testing.assert_array_equal(result, [0, 0])
        np.testing.assert_array_equal(parent.genome, [1, 0])
        self.assertEqual(random.call_count, 2)

    def test_mutation_produces_independent_individuals(self):
        for probability in (0, 1):
            with self.subTest(probability=probability):
                children = Mutate(probability).mutate(self.population)
                self.assertEqual(children.N, 2 * self.population.N)
                for index, parent in enumerate(self.population.generation):
                    first, second = children.generation[2 * index:2 * index + 2]
                    expected = parent.genome if probability == 0 else 1 - parent.genome
                    np.testing.assert_array_equal(first.genome, expected)
                    np.testing.assert_array_equal(second.genome, expected)
                    self.assertFalse(np.shares_memory(first.genome, second.genome))
                    self.assertFalse(np.shares_memory(first.genome, parent.genome))

    def test_evolution_mutates_then_selects_and_preserves_population(self):
        result = EvolutionStrategy(Truncate(0.5), Mutate(1)).evolve(self.population)
        self.assertEqual(result.N, self.population.N)
        self.assertEqual([i.fitness() for i in result.generation], [1, 1, 2, 2])

    def test_evolution_rejects_empty_result(self):
        with self.assertRaisesRegex(ValueError, 'No individuals selected'):
            EvolutionStrategy(Truncate(0), Mutate(0)).evolve(self.population)

    def test_experiment_records_and_plots_all_generations(self):
        for iterations in (0, 3):
            with self.subTest(iterations=iterations):
                experiment = Experiment(
                    EvolutionStrategy(Truncate(0.5), Mutate(0)),
                    self.population, iterations,
                )
                plots = [
                    (experiment.plot_fitness, 'avg_fitness'),
                    (experiment.plot_fitness_variace, 'fitness_variance'),
                    (experiment.plot_genome_variance, 'genome_variance'),
                ]
                for plot, _ in plots:
                    figure = Figure()
                    with self.assertRaisesRegex(ValueError, 'has not been run'):
                        plot(figure, figure.subplots())

                experiment.run_experiment()
                self.assertEqual(len(experiment.lineage_record), iterations + 1)
                self.assertTrue(all(g.N == self.population.N for g in experiment.lineage_record))
                for plot, metric in plots:
                    figure = Figure()
                    axes = figure.subplots()
                    plot(figure, axes)
                    np.testing.assert_array_equal(axes.lines[0].get_xdata(), range(iterations + 1))
                    np.testing.assert_allclose(
                        axes.lines[0].get_ydata(),
                        [getattr(g, metric)() for g in experiment.lineage_record],
                    )
                with self.assertRaisesRegex(ValueError, 'already run'):
                    experiment.run_experiment()


if __name__ == '__main__':
    unittest.main()
