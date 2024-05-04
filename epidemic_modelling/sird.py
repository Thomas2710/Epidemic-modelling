from typing import TypeAlias, Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult

Susceptible: TypeAlias = float
Infected: TypeAlias = float
Recovered: TypeAlias = float
Deaths: TypeAlias = float

WeekUpdate: TypeAlias = Tuple[Susceptible, Infected, Recovered, Deaths]


class SIRD:
    def __init__(self, s0, i0, r0, d0):
        self.susceptible: list[Susceptible] = [s0]
        self.infected: list[Infected] = [i0]
        self.recovered: list[Recovered] = [r0]
        self.deaths: list[Deaths] = [d0]
        self.delta_t: list[np.array] = []
        self.phi_t_q: list[np.array] = []

    def __call__(self, update: WeekUpdate):
        s, i, r, d = update
        s_prev, i_prev, r_prev, d_prev = (
            self.susceptible[-1],
            self.infected[-1],
            self.recovered[-1],
            self.deaths[-1],
        )

        # step 1: compute (t+1) - (t)
        ds = s - self.susceptible[-1]
        di = i - self.infected[-1]
        dr = r - self.recovered[-1]
        dd = d - self.deaths[-1]

        delta_t = np.array([ds, di, dr, dd])
        self.delta_t.append(delta_t)

        # step 2: compute prediction

        update_s = (s_prev * i_prev) / (s_prev + i_prev)

        phi_t_q = np.array(
            [
                [-update_s, 0, 0],
                [update_s, -i_prev, i_prev],
                [0, i_prev, 0],
                [0, 0, i_prev],
            ]
        )
        self.phi_t_q.append(phi_t_q)

        # step 3: minimize the fitness function
        initial_guess = np.array([10, 10, 10])
        solution: OptimizeResult = minimize(
            SIRD._fitness_function,
            initial_guess,
            args=(len(self.delta_t), (self.delta_t, self.phi_t_q)),
        )

        if solution.success:
            print("Optimization successful")
            print(solution.x)

        # step 4: update the state
        self.susceptible.append(s)
        self.infected.append(i)
        self.recovered.append(r)
        self.deaths.append(d)

    # TODO: change forget_rate to be a function of time
    @staticmethod
    def _fitness_function(
        params: Tuple,
        timesteps: int,
        history: Tuple[
            list[np.array],
            list[np.array],
        ],
        forget_rate: float = 0.2,
    ) -> float:
        retval = 0
        delta, phi = history
        for t in range(timesteps):
            delta_t, phi_t = delta[t], phi[t]
            val = np.linalg.norm(delta_t - np.dot(phi_t, params))
            val = val**2
            val *= forget_rate ** (timesteps - t)
            retval += val
        return retval / timesteps


def main():
    sird = SIRD(1000, 90, 30, 5)
    sird((900, 100, 60, 10))
