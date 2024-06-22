import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch


class TimeSeriesDataset:
    train_dataset = None
    test_dataset = None

    @staticmethod
    def load_data(config):
        """
        Take SIRD parameters solutions from csv and return list of lists

        @param: config file with filename
        @return: list of lists
        """
        file_path = getattr(config, "PRE", getattr(config, "NAME", None))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f"../../data/solutions/{file_path}.csv")
        sird_path = os.path.join(script_dir, "../../data/daily_processed.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        with open(file_path, "r") as solution_file:
            df_raw = pd.read_csv(solution_file)

        with open(sird_path, "r") as sird_file:
            df_sird = pd.read_csv(sird_file)

        # keep columns in this order [suscettibili,totale_positivi,dimessi_guariti,deceduti]
        df_sird = df_sird[
            ["suscettibili", "totale_positivi", "dimessi_guariti", "deceduti"]
        ]
        
        return df_raw.values.tolist(), df_sird.values.tolist()

    @staticmethod
    def preprocess_data(params, sird, input_length, target_length, offset=7):
        assert target_length == 1, "Target length must be 1 when working with LSTM"
        
        # LET'S ALWAYS CONSIDER THAT WE ARE SIMULATING 7 DAYS IN 7 DAYS
        # SO THAT DATA -> CONTAINS SIRD PARAMS FOR 1 WEEK
        # data -> [{beta,gamma,delta},...{beta, gamma, delta}]
        # Parameters dataset
        starting_params = [
            params[i : i + input_length] for i in range(len(params) - input_length - offset - 1)
        ]

        # SHOULD CONTAINS all the S,I,R,D real values day by day
        # SO THAT IT WILL CONTAINS: DAYS*SEGMENTS rows
        # 7 days considered for 4 segments => 28 rows
        # TODO: verify if it is doing so that
        original_sird = [
            sird[i : i + offset + 1] for i in range(len(params) - input_length - offset - 1)
        ]
        
        return starting_params, original_sird

    @staticmethod
    def split_data(starting_params, original_sird):
        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = train_test_split(
            starting_params,
            original_sird,
            test_size=0.3,
            random_state=42,
            shuffle=False,
        )
        
        if len(x_test) % 2 != 0:
            x_test = x_test[:-1]
            y_test = y_test[:-1]
        
        TimeSeriesDataset.train_dataset = TensorDataset(
            torch.tensor(x_train),
            torch.tensor(y_train),
        )
        TimeSeriesDataset.test_dataset = TensorDataset(
            torch.tensor(x_test),
            torch.tensor(y_test),
        )
