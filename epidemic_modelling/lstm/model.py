import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from epidemic_modelling.lstm.dataset import TimeSeriesDataset


class LSTMModel(pl.LightningModule):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=3):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.criterion = nn.MSELoss()
        self.h0 = None
        self.c0 = None
 
    def forward(self, input_seq):
        # shape of input_seq: (batch_size, seq_len, feature_size)
        lstm_out, _ = self.lstm(input_seq)
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        # print(predictions.shape)
        return predictions.unsqueeze(1)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        y_pred = self(sequences)
        loss = self.criterion(y_pred, labels)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        sequences, labels = batch
        y_pred = self(sequences)
        loss = self.criterion(y_pred, labels)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        train_dataset = TimeSeriesDataset.train_dataset
        return DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=2
        )
        

    def test_dataloader(self):
        test_dataset = TimeSeriesDataset.test_dataset
        return DataLoader(
            test_dataset, batch_size=8, shuffle=False, num_workers=2
        )
