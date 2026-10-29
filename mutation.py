from main import SelectionStrategy,MutationStrategy,EvolutionStrategy,Experiment,Generation,Individuel
from math import floor 
import numpy as np
from random import random

class Truncate(SelectionStrategy) :
    def __init__(self , prc : int):
        super().__init__()
        self.prc = prc

    def select(self, generation : Generation):
        indiviuels = generation.generation
        sorted_generation = sorted(indiviuels , key = lambda i : i.fitness() )
        treshhold = floor(self.prc  * generation.N)
        new_generation = Generation(sorted_generation[generation.N - treshhold : generation.N])
        return new_generation
    
def mutate_individuel (individuel : Individuel , p : int) :
    
    mutations = np.array([1 if random() < p else 0 for _ in range(individuel.G)], dtype=int)

    new_genome = np.bitwise_xor(mutations, individuel.genome)

    return new_genome
        

class Mutate(MutationStrategy) :
    def __init__(self, p ):
        super().__init__()
        if not (0<= p <= 1) :
            raise ValueError("Invalid probability input")
        self.p = p
    def mutate(self, generation : Generation):
        new_generation = []
        for i in generation.generation:
            new_individuel = mutate_individuel(i , self.p)
            # twice to keep the population number constant
            new_generation.append(Individuel(new_individuel))
            new_generation.append(Individuel(new_individuel)) 
        return Generation(new_generation)
