# Lambda = contact rate
# Gamma = recovery rate
# Mu = death rate
# Average mortality rate = Mu/Gamma
# R0 (contact number) = Lambda/(Gamma + Mu)

# dS/dt = -Lambda * S * I
# dI/dt = Lambda * S * I - (Gamma * I) - (Mu * I)
# dR/dt = Gamma * I
# dD/dt = Mu * I
# S + I + R + D = 1

# Infection over time
# I(t) is proprtional to exp(Lambda * ((S - 1/R0)*I))

from scipy.integrate import solve_ivp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

initial_conditions = {
    "population": 60000000,
    "cases": 3000,
    "deaths": 80,
    "recovered": 20,
}


class SIRD:
    def __init__(self, R0: float, M: float, P: int):
        # Model parameters

        # R0: Basic Reproductive Rate [people]
        self.R0 = R0
        # M: mortality rate ratio
        self.M = M
        # P: Average infectious period [days]
        self.P = P

    def dSdt(self, S: int, I: int, lamb: float):
        """
        Compute Susceptible parameter over time

        Parameters:
        ------------
        S: Susceptible population
        I: Infected population
        lamb: Contact rate

        Returns:
        ------------
        dSdt: Susceptible population over time
        """
        return -lamb * S * I

    def dIdt(self, S: int, I: int, lamb: float, gamma: float, mu: float):
        """
        Compute Infected parameter over time

        Parameters:
        ------------

        S: Susceptible population
        I: Infected population
        lamb: Contact rate
        gamma: Recovery rate
        mu: Death rate

        Returns:
        ------------
        dIdt: Infected population over time
        """
        return lamb * S * I - gamma * I - mu * I

    def dRdt(self, I: int, gamma: float):
        """
        Compute Recovered parameter over time

        Parameters:
        ------------
        I: Infected population
        gamma: Recovery rate

        Returns:
        ------------
        dRdt: Recovered population over time
        """
        return gamma * I

    def dDdt(self, I: int, mu: float):
        """
        Compute Deaths parameter over time

        Parameters:
        ------------
        I: Infected population
        mu: Death rate

        Returns:
        ------------
        dDdt: Deaths population over time
        """

        return mu * I

    def eqns(self, t: int, y: tuple, lamb: float, gamma: float, mu: float):
        S, I, R, D = y
        return [
            self.dSdt(S, I, lamb),
            self.dIdt(S, I, lamb, gamma, mu),
            self.dRdt(I, gamma),
            self.dDdt(I, mu),
        ]

    def setup(self, population: int, cases: int, recovered: int, deaths: int):
        # Compute initial values
        self.population = population
        initial_S = (population - cases) / population
        initial_R = recovered / population
        initial_D = deaths / population
        initial_I = 1 - initial_S - initial_R - initial_D
        self.y0 = [initial_S, initial_I, initial_R, initial_D]

        # Compute coefficients
        self.gamma = 1 / self.P
        self.mu = self.gamma * self.M
        self.lamb = self.R0 * (self.gamma + self.mu)

    def solve(self, initial_conditions: dict, time_frame: int = 300):
        """
        Solve the SIRD model

        Parameters:
        ------------
        initial_conditions: dict
            Dictionary containing initial conditions for the model
            keys: population, cases, recovered, deaths
        time_frame: int
            Number of days to run simulation for

        Returns:
        ------------
        self: SIRD
            Returns the instance of the class
        """

        self.setup(
            initial_conditions["population"],
            initial_conditions["cases"],
            initial_conditions["recovered"],
            initial_conditions["deaths"],
        )

        t_span = (
            0,
            time_frame,
        )  # tf is number of days to run simulation for, defaulting to 300

        self.soln = solve_ivp(
            self.eqns,
            t_span,
            self.y0,
            args=(self.lamb, self.gamma, self.mu),
            t_eval=np.linspace(0, time_frame, time_frame * 2),
        )
        return self
    
    def get_params(self):
        """
        Return the model parameters after a simulation has been run
        
        Returns:
        ------------
        dict: dictionary containing model parameters
        """
        params = self.soln.y[:, -1]
        return { "S": params[0], "I": params[1], "R": params[2], "D": params[3]}

    def plot(self, ax=None, susceptible=True):
        S, I, R, D = self.soln.y
        t = self.soln.t
        N = self.population

        print(f"For a population of {N} people, after {t[-1]:.0f} days there were:")
        print(f"{D[-1]*100:.1f}% total deaths, or {D[-1]*N:.0f} people.")
        print(f"{R[-1]*100:.1f}% total recovered, or {R[-1]*N:.0f} people.")
        print(
            f"At the virus' maximum {I.max()*100:.1f}% people were simultaneously infected, or {I.max()*N:.0f} people."
        )
        print(
            f"After {t[-1]:.0f} days the virus was present in less than {I[-1]*N:.0f} individuals."
        )

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_title("Covid-19 spread")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Number")
        if susceptible:
            ax.plot(t, S * N, label="Susceptible", linewidth=2, color="blue")
        ax.plot(t, I * N, label="Infected", linewidth=2, color="orange")
        ax.plot(t, R * N, label="Recovered", linewidth=2, color="green")
        ax.plot(t, D * N, label="Deceased", linewidth=2, color="black")
        ax.legend()

        return ax
