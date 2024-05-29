import functools
import os
import random
from time import time
from random import Random
import inspyred
import numpy as np
from inspyred.benchmarks import Benchmark
from inspyred import ec
import math
import pandas as pd
from inspyred.swarm import topologies

import inspyred.ec.terminators

SEED = 42
MAX_GENERATIONS = 100
POPULATION_SIZE = 100


def write_to_csv(idx, w, s, i, r, d, fitness):
    with open("../data/plot.csv", "a") as f:
        if f.tell() == 0:
            f.write(
                "idx,week_considered,susceptible,infected,recovered,death,fitness_value\n"
            )
        f.write(f"{idx},{w},{s},{i},{r},{d},{fitness}\n")


class MySIRD(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self.bounder = ec.Bounder([0.0] * self.dimensions, [1.0] * self.dimensions)
        self.maximize = False
        # self.global_optimum = [0 for _ in range(self.dimensions)]
        self.t = 0
        self.data = pd.read_csv("../data/processed.csv")
        self.population = 60_000_000
        random.seed = SEED

    def generator(self, random, args):
        # Generate an initial random candidate for each dimension
        x = [random.uniform(0.0, 1.0) for _ in range(self.dimensions)]
        return x

    def evaluator(self, candidates, args):
        # TODO: MSE and SIRD integration
        fitness = []

        # For the moment we are going to consider only the first 10 weeks
        for current_time in range(1, 10):
            week = current_time
            infected_t = self.data["totale_positivi"].iloc[current_time - 1]
            recovered_t = self.data["dimessi_guariti"].iloc[current_time - 1]
            deceased_t = self.data["deceduti"].iloc[current_time - 1]

            # initial conditions
            initial_susceptible = (
                self.population - infected_t - recovered_t - deceased_t
            )
            initial_infected = infected_t
            initial_recovered = recovered_t
            initial_deceased = deceased_t

            # print(
            #     f"At week:{week}\n\tSusceptible: {initial_susceptible}, Infected: {initial_infected}, Recovered: {initial_recovered}, Deceased: {initial_deceased}"
            # )

            computed_population = (
                initial_susceptible
                + initial_infected
                + initial_recovered
                + initial_deceased
            )
            match = self.population == computed_population
            assert match, "Error in the computation of the population!"

            # print(
            #     f"Total population: {self.population}, S+I+R+D: {computed_population}, Matching {self.population == computed_population}"
            # )

            # input()

            for idx, (beta, gamma, delta) in enumerate(candidates):
                # susceptible, infected, recovered, deceased of the current candidate
                current_susceptible = (
                    initial_susceptible - beta * initial_susceptible * initial_infected
                )
                current_infected = (
                    initial_infected
                    + beta * initial_susceptible * initial_infected
                    - gamma * initial_infected
                    - delta * initial_infected
                )
                current_recovered = initial_recovered + gamma * initial_infected
                current_deceased = initial_deceased + delta * initial_infected

                
                desired_susceptible = self.data["totale_positivi"].iloc[current_time]
                desired_infected = self.data["dimessi_guariti"].iloc[current_time]
                desired_recovered = self.data["deceduti"].iloc[current_time]
                desired_deceased = self.data["dimessi_guariti"].iloc[current_time]

                loss_susceptible = (current_susceptible - desired_susceptible) ** 2
                loss_infected = (current_infected - desired_infected) ** 2
                loss_recovered = (current_recovered - desired_recovered) ** 2
                loss_deceased = (current_deceased - desired_deceased) ** 2

                # Print losses obtained
                # print(
                #     f"Losses: S: {loss_susceptible}, I: {loss_infected}, R: {loss_recovered}, D: {loss_deceased}"
                # )

                loss_normalized = np.mean(
                    [loss_susceptible, loss_infected, loss_recovered, loss_deceased]
                )

                # print(f"Loss normalized: {loss_normalized}")

                write_to_csv(
                    idx,
                    week,
                    current_susceptible,
                    current_infected,
                    current_recovered,
                    current_deceased,
                    loss_normalized,
                )

                fitness.append(loss_normalized)

        # print(fitness)
        # input()
        return fitness

    @staticmethod
    def should_terminate(population, num_generations, num_evaluations, args):
        print(f"Generation # {num_generations} ...")
        return num_generations >= MAX_GENERATIONS


def main(prng=None, display=True):
    # If previous plots exist, remove them
    if os.path.exists("../data/plot.csv"):
        os.remove("../data/plot.csv")

    # Initialization of pseudorandom number generator
    if prng is None:
        prng = Random()
        prng.seed(SEED)

    # Defining the 3 parameters to optimize
    problem = MySIRD(3)
    ea = inspyred.swarm.PSO(prng)
    ea.terminator = MySIRD.should_terminate
    ea.topology = topologies.ring_topology

    final_pop = ea.evolve(
        generator=problem.generator,
        evaluator=problem.evaluator,
        pop_size=POPULATION_SIZE,
        bounder=problem.bounder,
        maximize=problem.maximize,
    )

    with open("../data/plot.csv", "r") as f:
        # print number of lines
        print(f"Entries generated in the csv log: {len(f.readlines())-1}")

    best = max(final_pop)

    if display:
        print(f"Best solution provided: {best}")
    
    # Write on csv the best solution
    with open("../data/best_solution.csv", "w") as f:
        f.write("beta,gamma,delta\n")
        f.write(f"{best.candidate[0]},{best.candidate[1]},{best.candidate[2]}\n")
    
    return ea


if __name__ == "__main__":
    main(display=True)
