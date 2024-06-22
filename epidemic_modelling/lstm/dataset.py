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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f"../../data/solutions/{config.NAME}.csv")
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
    def preprocess_data(data, sird, input_length, target_length, offset=1):
        # LET'S ALWAYS CONSIDER THAT WE ARE SIMULATING 7 DAYS IN 7 DAYS
        # SO THAT DATA -> CONTAINS SIRD PARAMS FOR 1 WEEK
        # data -> [{beta,gamma,delta},...{beta, gamma, delta}]
        # Parameters dataset

        # TODO: TRY 1 week in -> 1 week out
        sequences = [
            data[i : i + input_length] for i in range(len(data) - input_length - offset)
        ]
        labels = [
            data[i + input_length : i + input_length + target_length]
            for i in range(len(data) - input_length - offset)
        ]

        # SHOULD CONTAINS all the S,I,R,D real values day by day
        # SO THAT IT WILL CONTAINS: DAYS*SEGMENTS rows
        # 7 days considered for 4 segments => 28 rows
        # TODO: verify if it is doing so that
        sird_initial = [
            sird[i + input_length : i + input_length + target_length]
            for i in range(len(data) - input_length - offset)
        ]
        sird_final = [
            sird[i + input_length + offset : i + input_length + target_length + offset]
            for i in range(len(data) - input_length - offset)
        ]
        return sequences, labels, sird_initial, sird_final

    @staticmethod
    def split_data(sequences, labels, sird_initial, sird_final):
        (
            x_train,
            x_test,
            y_train,
            y_test,
            sird_initial_train,
            sird_initial_test,
            sird_final_train,
            sird_final_test,
        ) = train_test_split(
            sequences,
            labels,
            sird_initial,
            sird_final,
            test_size=0.3,
            random_state=42,
            shuffle=False,
        )

        TimeSeriesDataset.train_dataset = TensorDataset(
            torch.tensor(x_train),
            torch.tensor(y_train),
            torch.tensor(sird_initial_train),
            torch.tensor(sird_final_train),
        )
        TimeSeriesDataset.test_dataset = TensorDataset(
            torch.tensor(x_test),
            torch.tensor(y_test),
            torch.tensor(sird_initial_test),
            torch.tensor(sird_final_test),
        )
