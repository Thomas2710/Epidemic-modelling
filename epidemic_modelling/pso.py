from time import time
from random import Random
import inspyred
from inspyred.benchmarks import Benchmark
from inspyred import ec
import math
import pandas as pd
from inspyred.swarm import topologies

class MySIRD(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self.bounder = ec.Bounder([0.0] * self.dimensions, [1.0] * self.dimensions)
        self.maximize = False
        self.global_optimum = [0 for _ in range(self.dimensions)]
        self.t = 0
        self.data = pd.read_csv("data/processed.csv")
        self.population = 60_000_000

    def generator(self, random, args):
        x = [random.uniform(0.0, 1.0) for _ in range(self.dimensions)]
        print(x)
        return x

    def evaluator(self, candidates, args):
        # TODO: MSE and SIRD integration
        fitness = []
        for c in candidates:
            beta, gamma, delta = c
            si = self.population - self.data["totale_positivi"].iloc[self.t]
            ii = self.data["totale_positivi"].iloc[self.t] - self.data["dimessi_guariti"].iloc[self.t] - self.data["deceduti"].iloc[self.t]
            ri = self.data["dimessi_guariti"].iloc[self.t]
            di = self.data["deceduti"].iloc[self.t]

            sc = si - beta * si * ii
            ic = ii + beta * si * ii - gamma * ii - delta * ii
            rc = ri + gamma * ii
            dc = di + delta * ii

            # print(f"Actual S: {sc}, Expected S: {self.population - self.data['totale_positivi'].iloc[self.t + 1]}")
            # print(f"Actual I: {ic}, Expected I: {self.data['totale_positivi'].iloc[self.t + 1] - self.data['dimessi_guariti'].iloc[self.t + 1] - self.data['deceduti'].iloc[self.t + 1]}")
            # print(f"Actual R: {rc}, Expected R: {self.data['dimessi_guariti'].iloc[self.t + 1]}")
            # print(f"Actual D: {dc}, Expected D: {self.data['deceduti'].iloc[self.t + 1]}")
            # print("--------------------")

            fitness.append((ic - (self.data["totale_positivi"].iloc[self.t + 1] - self.data["dimessi_guariti"].iloc[self.t + 1] - self.data["deceduti"].iloc[self.t + 1]))**2 + (rc - self.data["dimessi_guariti"].iloc[self.t + 1])**2 + (dc - self.data["deceduti"].iloc[self.t + 1])**2)

        # self.t += 1
        print(min(fitness))
        return fitness
    
    @staticmethod
    def should_terminate(population, num_generations, num_evaluations, args):
        return num_evaluations >= 50000
        

def main(prng=None, display=True):
    if prng is None:
        prng = Random()
        prng.seed(time()) 
    
    problem = MySIRD(3)
    ea = inspyred.swarm.PSO(prng)
    ea.terminator = MySIRD.should_terminate
    ea.topology = topologies.ring_topology
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator, 
                          pop_size=100,
                          bounder=problem.bounder,
                          maximize=problem.maximize,
                          max_evaluations=50000, 
                          neighborhood_size=9)

    if display:
        best = max(final_pop) 
        print('Best Solution: \n{0}'.format(str(best)))
    return ea

if __name__ == '__main__':
    main(display=True)