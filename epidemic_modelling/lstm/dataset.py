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
        '''
        Take SIRD parameters solutions from csv and return list of lists

        @param: config file with filename
        @return: list of lists 
        '''
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f"../../data/solutions/{config.NAME}.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        with open(file_path, "r") as solution_file:
            df_raw = pd.read_csv(solution_file)
        return df_raw.values.tolist()

    @staticmethod
    def preprocess_data(data, input_length, target_length):
        sequences = [
            data[i : i + input_length]
            for i in range(len(data) - input_length)
        ]
        labels = [
            data[i + input_length: i+input_length+target_length] for i in range(len(data) - input_length)
        ]
        return sequences, labels

    @staticmethod
    def split_data(sequences, labels):
        x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.3, random_state=42)

        TimeSeriesDataset.train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        TimeSeriesDataset.test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
        