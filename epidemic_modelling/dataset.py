import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, config="time_varying"):
        self.config = config
        self.sequences = self._load_data()
        self.bidirectional = False
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.float32))

    def _load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f"../data/processed/{self.config}.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        with open(file_path, 'r') as solution_file:
            df_raw = pd.read_csv(solution_file)
        return list(df_raw.to_records(index=False))

def test():
    dataset = TimeSeriesDataset()
    print(len(dataset))
    print(dataset[0])
    
if __name__ == "__main__":
    test()
