import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.criterion = nn.MSELoss()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        y_pred = self(sequences)
        loss = self.criterion(y_pred, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        y_pred = self(sequences)
        loss = self.criterion(y_pred, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def train_dataloader(self):
        train_dataset = LSTMData()
        return DataLoader()

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()
    