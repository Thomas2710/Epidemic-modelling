import os
import random

import click
import numpy as np
import pandas as pd
import torch
import inspyred
from inspyred.swarm import topologies
from inspyred import ec
from inspyred.ec.emo import Pareto
from inspyred.benchmarks import Benchmark

from epidemic_modelling.lstm.dataset import TimeSeriesDataset
from epidemic_modelling.lstm.main import train
from epidemic_modelling.lstm.model import LSTMModel
from epidemic_modelling.sird_base_model import SIRD
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

min_conv = []
max_conv = []
current_seg = 0


# Function to convert a number to engineering notation
def to_eng_notation(value):
    if value == 0:
        return "0"

    exponent = int(math.floor(math.log10(abs(value)) / 3) * 3)
    scaled_value = value / (10**exponent)

    return f"{scaled_value:.3g}e{exponent:+}"


class BaseConfig:
    def __init__(self) -> None:
        self.SEED = 42
        self.MAX_GENERATIONS = 1e2
        self.POPULATION_SIZE = 3e2
        self.LAG = 0

        self.PARAMS_THRESHOLD = 0.99
        self.FACTOR_LOWER_BOUND = 0.001
        self.FACTOR_UPPER_BOUND = 1.0

        self.weight_S = 0.8
        self.weight_I = 5
        self.weight_R = 2
        self.weight_D = 3

        self.cognitive_rate = 2.1
        self.social_rate = 1.2
        self.inertia = 0.65
        self.neighborhood = 5


class BaselineConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.SEGMENTS = 1
        self.NAME = "baseline"
        self.DAYS = 56


class TimeVaryingConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.SEGMENTS = 8
        self.NAME = "time_varying"
        self.DAYS = 7


class LSTMConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.EPOCHS = 200
        self.LOG_EVERY_N_STEPS = 5
        self.SEGMENTS = 10
        self.PRE = "time_varying_pre_lstm"
        self.POST = "time_varying_post_lstm"
        self.DAYS = 7
        self.IN_WEEKS = 1
        self.OUT_WEEKS = 1


class ParetoLoss(Pareto):
    def __init__(self, pareto, args):
        """edit this function to change the way that multiple objectives
        are combined into a single objective

        """

        Pareto.__init__(self, pareto)
        if "fitness_weights" in args:
            weights = np.asarray(args["fitness_weights"])
        else:
            weights = np.asarray([1 for _ in pareto])

        self.fitness = sum(np.asarray(pareto * weights))

    def __lt__(self, other):
        return self.fitness < other.fitness


class MyPSO(Benchmark):
    def __init__(
        self,
        dimensions=3,
        config: BaseConfig = None,
        best_initial_condition: dict = None,
    ):
        Benchmark.__init__(self, dimensions)
        self.config = config
        self.bounder = ec.Bounder(
            [self.config.FACTOR_LOWER_BOUND] * self.dimensions,
            self.config.FACTOR_UPPER_BOUND * self.dimensions,
        )
        self.maximize = False
        # Absolute path of the data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "../data/daily_processed.csv")
        self.data = pd.read_csv(filepath)
        self.population = 60_000_000
        self.epoch = 0
        random.seed = self.config.SEED
        if best_initial_condition is not None:
            self.best_initial_condition = best_initial_condition

    def generator(self, random, args):
        # Generate an initial random candidate for each dimension
        x = [
            random.uniform(
                self.config.FACTOR_LOWER_BOUND, self.config.FACTOR_UPPER_BOUND
            )
            for _ in range(self.dimensions)
        ]
        return x

    def setup(self):
        if current_seg == 0:
            self.initial_conds, _ = self.get_sird_from_data(
                self.config.LAG, self.config.DAYS + self.config.LAG, self.population
            )
        else:
            self.initial_conds = self.best_initial_condition
        _, self.future_conds = self.get_sird_from_data(
            self.config.LAG + 1,
            self.config.DAYS + self.config.LAG + 1,
            self.population,
        )

    def evaluator(self, candidates, args):
        fitness = []

        future_params = [
            self.future_conds["initial_S"],
            self.future_conds["initial_I"],
            self.future_conds["initial_R"],
            self.future_conds["initial_D"],
        ]
        partial_losses = []
        args = {}
        args["fitness_weights"] = [
            self.config.weight_S,
            self.config.weight_I,
            self.config.weight_R,
            self.config.weight_D,
        ]
        for idx, (beta, gamma, delta) in enumerate(candidates):
            model = SIRD(beta=beta, gamma=gamma, delta=delta)
            # solve
            days = self.config.DAYS
            # pickup GT
            model.solve(self.initial_conds, days)
            # Values obtained
            computed_S, computed_I, computed_R, computed_D, sum_params = (
                model.get_sird_values().values()
            )
            current_params = [computed_S, computed_I, computed_R, computed_D]
            # Check if the sum of the parameters is valid
            assert (
                sum_params.all() >= self.config.PARAMS_THRESHOLD
            ), f"Sum of parameters is less than {self.config.PARAMS_THRESHOLD}"

            # compute loss
            losses = model.compute_loss(current_params, future_params, loss="RMSE")
            partial_losses.append(losses)

            # loss_normalized = np.mean(losses)
            fitness.append(ParetoLoss(losses, args=args))

        min_conv.append(min(fitness).fitness)
        max_conv.append(max(fitness).fitness)
        self.epoch += 1
        return fitness

    def should_terminate(self, population, num_generations, num_evaluations, args):
        return num_generations >= self.config.MAX_GENERATIONS

    def get_sird_from_data(self, start_week: int, end_week: int, population: int):
        start_week = start_week - 1 if start_week > 0 else 0
        infected_t = (
            self.data["totale_positivi"]
            .iloc[start_week:end_week]
            .to_numpy()
            .astype(float)
        )
        recovered_t = (
            self.data["dimessi_guariti"]
            .iloc[start_week:end_week]
            .to_numpy()
            .astype(float)
        )
        deceased_t = (
            self.data["deceduti"].iloc[start_week:end_week].to_numpy().astype(float)
        )
        susceptible_t = (
            self.data["suscettibili"].iloc[start_week:end_week].to_numpy().astype(float)
        )
        all_conds = {
            "population": population,
            "initial_I": infected_t,
            "initial_R": recovered_t,
            "initial_D": deceased_t,
            "initial_S": susceptible_t,
        }
        initial_conds = {
            "population": population,
            "initial_I": infected_t[0],
            "initial_R": recovered_t[0],
            "initial_D": deceased_t[0],
            "initial_S": susceptible_t[0],
        }
        return initial_conds, all_conds

    def save_best_solution(self, final_pop, display=True):
        file_path = getattr(self.config, "PRE", getattr(self.config, "NAME", None))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(script_dir, "../data/solutions")):
            os.makedirs(os.path.join(script_dir, "../data/solutions"))
        best_solution_filepath = os.path.join(
            script_dir, f"../data/solutions/{file_path}.csv"
        )

        # best = min(final_pop, key=lambda x: x.fitness)
        best = min(final_pop, key=lambda x: x.fitness)

        with open(best_solution_filepath, "a+") as f:
            if f.tell() == 0:
                f.write("beta,gamma,delta\n")
            f.write(f"{best.candidate[0]},{best.candidate[1]},{best.candidate[2]}\n")

        return best.candidate[0], best.candidate[1], best.candidate[2]


def clean_paths(config):
    file_path = getattr(config, "PRE", getattr(config, "NAME", None))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(script_dir, "../data/solutions")):
        os.makedirs(os.path.join(script_dir, "../data/solutions"))
    best_solution_filepath = os.path.join(
        script_dir, f"../data/solutions/{file_path}.csv"
    )
    if os.path.exists(best_solution_filepath):
        os.remove(best_solution_filepath)


def save_post(config, params_collection):
    file_path = config.POST
    script_dir = os.path.dirname(os.path.abspath(__file__))
    post_file_path = os.path.join(script_dir, f"../data/solutions/{file_path}.csv")
    if os.path.exists(post_file_path):
        os.remove(post_file_path)
    # set header to beta, gamma, delta
    params_collection.to_csv(
        post_file_path, index=False, header=["beta", "gamma", "delta"]
    )


@click.command()
@click.option("--display", default=True, is_flag=True, help="Display the best solution")
@click.option(
    "--time-varying",
    default=False,
    is_flag=True,
    help="Run the baseline with time-varying parameters",
)
@click.option(
    "--lstm",
    default=False,
    is_flag=True,
    help="Run the baseline with LSTM parameters",
)
@click.option("--prng", default=None, help="Seed for the pseudorandom number generator")
def main(display, time_varying, lstm, prng):
    if time_varying:
        config = TimeVaryingConfig()
    elif lstm:
        config = LSTMConfig()
    else:
        config = BaselineConfig()

    print(f"Running {config.__class__.__name__} configuration")
    clean_paths(config)

    best_initial_conditions = None
    for seg in tqdm(range(config.SEGMENTS), unit="Segment", position=0, leave=True):
        global current_seg
        current_seg = seg
        problem = MyPSO(3, config, best_initial_conditions)
        problem.setup()

        # Initialization of pseudorandom number generator
        if prng is None:
            prng = random.Random()
            prng.seed(config.SEED)

        # Defining the 3 parameters to optimize
        ea = inspyred.swarm.PSO(prng)
        ea.terminator = problem.should_terminate
        ea.topology = topologies.star_topology
        ea.social_rate = config.social_rate
        ea.cognitive_rate = config.cognitive_rate
        ea.inertia = config.inertia

        final_pop = ea.evolve(
            generator=problem.generator,
            evaluator=problem.evaluator,
            pop_size=problem.config.POPULATION_SIZE,
            bounder=problem.bounder,
            maximize=problem.maximize,
            social_rate=config.social_rate,
            cognitive_rate=config.cognitive_rate,
            inertia=config.inertia,
            neighborhood_size=config.neighborhood,
        )

        best = problem.save_best_solution(final_pop, display)
        # RUN SIRD with best parameters in that specific timeframe
        sird = SIRD(beta=best[0], gamma=best[1], delta=best[2])
        starting_point, _ = problem.get_sird_from_data(
            problem.config.LAG,
            problem.config.DAYS + problem.config.LAG,
            problem.population,
        )
        sird.solve(starting_point, problem.config.DAYS)
        computed_S, computed_I, computed_R, computed_D, sum_params = (
            sird.get_sird_values().values()
        )
        best_initial_conditions = {
            "population": problem.population,
            "initial_I": computed_I[-1],
            "initial_R": computed_R[-1],
            "initial_D": computed_D[-1],
            "initial_S": computed_S[-1],
        }
        # Plot the fitness value for each generation to see when it converges
        # check if convergence folder is present

        convergence_folder = f"cr_{config.cognitive_rate}_sr_{config.social_rate}_in_{config.inertia}_nh_{config.neighborhood}"
        if not os.path.exists(
            os.path.join(os.getcwd(), "convergence", convergence_folder)
        ):
            os.makedirs(os.path.join(os.getcwd(), "convergence", convergence_folder))
        start = int(seg * config.MAX_GENERATIONS)
        end = int(1 + (seg + 1) * config.MAX_GENERATIONS)
        x_values = range(0, int(config.MAX_GENERATIONS) + 1)
        # Plotting the numbers with customizations
        # Create the primary axis
        fig, ax1 = plt.subplots()

        # Plot the first dataset on the primary y-axis
        (line1,) = ax1.plot(
            x_values,
            min_conv[start:end],
            linestyle="--",
            color="b",
            label="MinF Segment " + str(seg + 1),
        )
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Min Fitness", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        plt.legend()
        # Create the secondary y-axis
        ax2 = ax1.twinx()
        (line2,) = ax2.plot(
            x_values,
            max_conv[start:end],
            linestyle="--",
            color="r",
            label="MaxF Segment " + str(seg + 1),
        )
        ax2.set_ylabel("Max Fitness", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        # Combine legends from both axes
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right")

        params_title = f"\nWeights = [S: {config.weight_S}, I: {config.weight_I}, R: {config.weight_R}, D: {config.weight_D}]\nPopulation: {config.POPULATION_SIZE}  Neighbourood: {config.neighborhood}\nCognitive Rate: {config.cognitive_rate}\nSocial Rate: {config.social_rate}\nInertia: {config.inertia}\n"
        max_min_desc = f"Max Fitness: {to_eng_notation(min(max_conv[start:end]))} Min Fitness: {to_eng_notation(min(min_conv[start:end]))}"
        plt.title(
            "\nPlot of Fitness convergence\n" + params_title + "\n" + max_min_desc
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                os.getcwd(), "convergence", convergence_folder, str(seg) + "_conv.png"
            )
        )
        config.LAG += config.DAYS

    if lstm:
        train(config, weeks_limit=config.SEGMENTS)
        model = LSTMModel.load_from_checkpoint("lstm_model.ckpt")
        model.eval()
        params, sird = TimeSeriesDataset.load_data(config)
        params = params[0]
        sird = sird[0]
        # add row with params and title beta, gamma, delta
        params_collection = pd.DataFrame(params).T
        # run inference
        for i in range(14):
            sird_tensor = torch.tensor(sird, dtype=torch.float32).unsqueeze(0)
            if len(sird_tensor.size()) < 3:
                sird_tensor = sird_tensor.unsqueeze(0)

            params_tensor = (
                torch.tensor(params, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            # run inference
            sird, params = model(params_tensor, sird_tensor)
            params = params.squeeze().detach().tolist()
            sird = sird[-1].squeeze().detach().tolist()
            params_collection = pd.concat([params_collection, pd.DataFrame(params).T])

        # save output in time_varying_post_lstm.csv
        save_post(config, params_collection)


if __name__ == "__main__":
    main()
