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
MAX_GENERATIONS = 1e3
POPULATION_SIZE = 1e3
WEEKS = 10
LAG = 30
DAYS = 70
PARAMS_THRESHOLD = 0.99
FACTOR_LOWER_BOUND = 0.001
FACTOR_UPPER_BOUND = 1.0

def write_to_csv(idx, w, s, i, r, d, fitness):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_filepath = os.path.join(script_dir, "../data/plot.csv")
    with open(plot_filepath, "a") as f:
        if f.tell() == 0:
            f.write(
                "idx,week_considered,susceptible,infected,recovered,death,fitness_value\n"
            )
        f.write(f"{idx},{w},{s},{i},{r},{d},{fitness}\n")

def get_sird_from_data(data: pd.DataFrame, start_week: int, end_week: int, population: int):
    infected_t = data["totale_positivi"].iloc[start_week:end_week].to_numpy().astype(float)
    recovered_t = data["dimessi_guariti"].iloc[start_week:end_week].to_numpy().astype(float)
    deceased_t = data["deceduti"].iloc[start_week:end_week].to_numpy().astype(float)
    susceptible_t = data['suscettibili'].iloc[start_week:end_week].to_numpy().astype(float)
    all_conds = {"population": population, "initial_I": infected_t, "initial_R": recovered_t, "initial_D": deceased_t, "initial_S": susceptible_t}
    initial_conds = {"population": population, "initial_I": infected_t[0], "initial_R": recovered_t[0], "initial_D": deceased_t[0], "initial_S": susceptible_t[0]}
    return initial_conds, all_conds

class MySIRD(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self.bounder = ec.Bounder([FACTOR_LOWER_BOUND] * self.dimensions, FACTOR_UPPER_BOUND * self.dimensions)
        self.maximize = False
        # self.global_optimum = [0 for _ in range(self.dimensions)]
        # self.t = 0
        # Absolute path of the data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "../data/daily_processed.csv")
        self.data = pd.read_csv(filepath)
        self.population = 60_000_000
        self.epoch = 0
        random.seed = SEED

    def generator(self, random, args):
        # Generate an initial random candidate for each dimension
        x = [random.uniform(FACTOR_LOWER_BOUND, FACTOR_UPPER_BOUND) for _ in range(self.dimensions)]
        return x

    def get_ird(self):
        infected = self.data["totale_positivi"].values[:]
    def evaluator(self, candidates, args):
        # TODO: MSE and SIRD integration


        fitness = []

        # For the moment we are going to consider only the first 10 weeks
        initial_conds, _ = get_sird_from_data(self.data, LAG, DAYS + LAG, self.population)
        _, future_conds = get_sird_from_data(self.data, LAG+1, DAYS + LAG + 1, self.population)
        future_params = [future_conds['initial_S'], future_conds['initial_I'], future_conds['initial_R'], future_conds['initial_D']]
    

        for idx, (beta, gamma, delta) in tqdm(enumerate(candidates)):
            model = SIRD(beta = beta, gamma = gamma, delta = delta)
            # solve
            days = DAYS
            # pickup GT
            model.solve(initial_conds, days)
            # Values obtained
            computed_S, computed_I, computed_R, computed_D, sum_params = model.get_sird_values().values()
            current_params = [computed_S, computed_I, computed_R, computed_D]
            # Check if the sum of the parameters is valid
            assert sum_params >= PARAMS_THRESHOLD, f"Sum of parameters is less than {PARAMS_THRESHOLD}"

            # compute loss
            losses = model.compute_loss(current_params, future_params, loss="MSE")

            # Print losses obtained
            # print(
            #     f"Losses: S: {loss_susceptible}, I: {loss_infected}, R: {loss_recovered}, D: {loss_deceased}"
            # )

            loss_normalized = np.mean(losses)

            # print(f"Loss normalized: {loss_normalized}")
            sird_vals = model.get_sird_values()
            write_to_csv(
                idx,
                self.epoch,
                sird_vals["S"],
                sird_vals["I"],
                sird_vals["R"],
                sird_vals["D"],
                loss_normalized,
            )

            fitness.append(loss_normalized)
            # print(f"\nFitness: {loss_normalized}\n")
        self.epoch += 1
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

    final_pop = ea.evolve(
        generator=problem.generator,
        evaluator=problem.evaluator,
        pop_size=POPULATION_SIZE,
        bounder=problem.bounder,
        maximize=problem.maximize,
    )

    # print(f"Final pop attributes: {final_pop}")
    with open(plot_filepath, "r") as f:
        # print number of lines
        print(f"Entries generated in the csv log: {len(f.readlines())-1}")

    best = max(final_pop, key=lambda x: x.fitness)

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
