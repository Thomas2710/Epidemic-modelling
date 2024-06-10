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
from sird_base_model import SIRD
from tqdm import tqdm

import inspyred.ec.terminators

SEED = 42
MAX_GENERATIONS = 5
POPULATION_SIZE = 100
WEEKS = 10

def write_to_csv(idx, w, s, i, r, d, fitness):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_filepath = os.path.join(script_dir, "../data/plot.csv")
    with open(plot_filepath, "a") as f:
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
        # self.t = 0
        # Absolute path of the data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "../data/processed.csv")
        self.data = pd.read_csv(filepath)
        self.population = 60_000_000
        random.seed = SEED

    def generator(self, random, args):
        # Generate an initial random candidate for each dimension
        x = [random.uniform(0.0, 1.0) for _ in range(self.dimensions)]
        return x

    def get_ird(self):
        infected = self.data["totale_positivi"].values[:]
    def evaluator(self, candidates, args):

        fitness = []
        time_frame = args.get("time_frame", 1)
        # For the moment we are going to consider only the first 10 weeks
        for current_time in range(0, 10):
            week = current_time
            infected_t = self.data["totale_positivi"].iloc[current_time]
            recovered_t = self.data["dimessi_guariti"].iloc[current_time]
            deceased_t = self.data["deceduti"].iloc[current_time]

            model = SIRD(2, 0, 5.1)   
            # print(
            #     f"At week:{week}\n\tSusceptible: {initial_susceptible}, Infected: {initial_infected}, Recovered: {initial_recovered}, Deceased: {initial_deceased}"
            # )

            # print(
            #     f"Total population: {self.population}, S+I+R+D: {computed_population}, Matching {self.population == computed_population}"
            # )

            # input()

            for idx, (beta, gamma, delta) in tqdm(enumerate(candidates)):
                init_conditions = {'initial_S': self.population, 'initial_I': infected_t, 'initial_R': recovered_t, 'initial_D': deceased_t}
                #time_frame is the amount of weeks ahead we want to compute the parameters
                model.setup(**init_conditions)
                loss_susceptible, loss_infected, loss_recovered, loss_deceased = model.compute_loss((beta, gamma, delta), time_frame=time_frame)
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
                    model.get_params()['S'],
                    model.get_params()['I'],
                    model.get_params()['R'],
                    model.get_params()['D'],
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
    # Get the absolute path of the data file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_filepath = os.path.join(script_dir, "../data/plot.csv")

    if os.path.exists(plot_filepath):
        os.remove(plot_filepath)

    # Initialization of pseudorandom number generator
    if prng is None:
        prng = Random()
        prng.seed(SEED)

    # Defining the 3 parameters to optimize
    problem = MySIRD(3)
    ea = inspyred.swarm.PSO(prng)
    ea.terminator = MySIRD.should_terminate
    ea.topology = topologies.ring_topology

    time_frame = 1

    final_pop = ea.evolve(
        generator=problem.generator,
        evaluator=problem.evaluator,
        pop_size=POPULATION_SIZE,
        bounder=problem.bounder,
        maximize=problem.maximize,
        time_frame = time_frame,
    )

    with open(plot_filepath, "r") as f:
        # print number of lines
        print(f"Entries generated in the csv log: {len(f.readlines())-1}")

    best = max(final_pop)

    if display:
        print(f"Best solution provided: {best}")
    
    # Write on csv the best solution
    best_solution_filepath = os.path.join(script_dir, "../data/best_solution.csv")
    with open(best_solution_filepath, "w") as f:
        f.write("beta,gamma,delta\n")
        f.write(f"{best.candidate[0]},{best.candidate[1]},{best.candidate[2]}\n")
    
    return ea


if __name__ == "__main__":
    main(display=True)
