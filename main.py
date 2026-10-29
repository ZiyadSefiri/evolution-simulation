import numpy as np
import matplotlib.pyplot as plt
from random import randbytes
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class Individuel:
    def __init__(self, genome: np.array):
        if any(bit not in (0, 1) for bit in genome):
            raise ValueError("Genome must contain only 0 and 1")

        self.genome = genome.copy()
        self.G = len(genome)

    def fitness(self) -> int:
        return self.genome.sum()
    
class Generation :
    def __init__(self  , generation : list[Individuel] ) :
        self.generation = generation
        self.N =  len(generation)
    def avg_fitness (self) :
        return np.array([individual.fitness() for individual in self.generation]).mean()
    def fitness_variance (self) :
        return np.array([individual.fitness() for individual in self.generation]).var()
    def genome_variance  (self) :
        return np.array([individual.genome for individual in self.generation]).var(axis=0).mean()
    
from abc import ABC,abstractmethod

class SelectionStrategy (ABC) :
    #the selction happens after mutation
    
    @abstractmethod
    def select(self , generation:Generation)->  Generation :
        pass

class MutationStrategy (ABC) :
    #each parent produces 2 identical children so as the number of the population stays constant
    @abstractmethod
    def mutate(self , generation:Generation)->  Generation :
        pass

class EvolutionStrategy:
    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        mutation_strategy: MutationStrategy
    ):
        self.selection_strategy = selection_strategy
        self.mutation_strategy = mutation_strategy

    def evolve(self, generation: Generation) -> Generation:
        mutated = self.mutation_strategy.mutate(generation)
        selected = self.selection_strategy.select(mutated)

        if selected.N == 0:
            raise ValueError("No individuals selected")

        return selected

class Experiment:
    def __init__(
        self,
        evolution_strategy: EvolutionStrategy,
        starting_generation: Generation,
        number_of_iterations: int
    ):
        self.evolution_strategy = evolution_strategy
        self.starting_generation = starting_generation
        self.number_of_iterations = number_of_iterations

        self.has_run_experiment = False
        self.lineage_record :list [Generation]= []

    def run_experiment(self):
        if self.has_run_experiment :
            raise ValueError("Experiment has been already run")
        current = self.starting_generation

        for _ in range(self.number_of_iterations):
            self.lineage_record.append(current)
            current = self.evolution_strategy.evolve(current)

        self.lineage_record.append(current)
        self.has_run_experiment = True
    
    def plot_fitness(self , fig : Figure , ax : Axes) :
        if not self.has_run_experiment :
            raise ValueError("Experiment has not been run")
        X = list(range(len(self.lineage_record)))
        Y = [generation.avg_fitness() for generation in self.lineage_record ]

        ax.plot(X,Y)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average Fitness")


    def plot_fitness_variace(self , fig : Figure , ax : Axes) :
        if not self.has_run_experiment :
            raise ValueError("Experiment has not been run")
        X = list(range(len(self.lineage_record)))
        Y = [generation.fitness_variance() for generation in self.lineage_record ]

        ax.plot(X,Y)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Variance")
        

    def plot_genome_variance(self, fig : Figure , ax : Axes) :
        if not self.has_run_experiment :
            raise ValueError("Experiment has not been run")
        X = list(range(len(self.lineage_record)))
        Y = [generation.genome_variance() for generation in self.lineage_record ]

        ax.plot(X,Y)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Genome Variance")
        


