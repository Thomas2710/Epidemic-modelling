import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchdiffeq import odeint

from epidemic_modelling.lstm.dataset import TimeSeriesDataset


class LSTMModel(pl.LightningModule):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=3, offset=7):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.offset = offset
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sird = SIRD(60e6)
        self.criterion = nn.MSELoss()

    def forward(self, starting_params, original_sird):
        # TODO:
        # load params beta, gamma,delta
        # RUN SIRD FOR 1 WEEK
        # Compare computed S,I,R,D values with the real ones

        # shape of starting_params: (batch_size, seq_len, feature_size)

        lstm_out, _ = self.lstm(starting_params)
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        self.sird.setup(predictions[:, 0], predictions[:, 1], predictions[:, 2])        
        # original_sird[:,0,:] -> starting SIRD values
                    
        sol = odeint(
            self.sird, original_sird[:,0,:], torch.linspace(1, self.offset, self.offset + 1)
        )
        # TODO: return all solutions
        return sol, predictions

        # runno il sird con questi parametri
        # calcolo la loss di S I R D?

        # print(predictions.shape)
        # return predictions.unsqueeze(1)

    def training_step(self, batch, batch_idx):
        starting_params, original_sird = batch
        y_pred, _ = self(starting_params, original_sird.float())
        # shape 0 is batch size
        # shape 1 is the number of days (1: is to skip the first day that has the same values of the original SIRD)
        # shape 2 is the number of features (S,I,R,D) (1: is to skip S)
        loss = self.criterion(y_pred[:, :, 1:], original_sird[:, :, 1:].float())
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        starting_params, original_sird = batch
        y_pred, _ = self(starting_params, original_sird.float())
        loss = self.criterion(y_pred[:, 1:, 1:], original_sird[:, 1:, 1:].float())
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)

    def train_dataloader(self):
        train_dataset = TimeSeriesDataset.train_dataset
        return DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    def test_dataloader(self):
        test_dataset = TimeSeriesDataset.test_dataset
        return DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)


class SIRD(nn.Module):
    def __init__(self, population):
        super(SIRD, self).__init__()
        self.population = population
        self.beta = None
        self.gamma = None
        self.delta = None

    def setup(self, beta, gamma, delta):
        self.beta = beta.unsqueeze(1)
        self.gamma = gamma.unsqueeze(1)
        self.delta = delta.unsqueeze(1)

    def forward(self, t, y):
        return self._get_eqns(y)

    def _get_eqns(self, y):
        # shape is [8,4], i want to split it into 4 tensors (innermost dimension)
        S, I, R, D = torch.split(y, 1, dim=1)
        N = self.population
        dSdt = -self.beta * S * I / N
        dIdt = (self.beta * S * I - self.gamma * I - self.delta * I) / N
        dRdt = self.gamma * I / N
        dDdt = self.delta * I / N
        # convert [8,1][8,1][8,1][8,1] to [8,4]
        stack = torch.stack([dSdt, dIdt, dRdt, dDdt], dim=1).squeeze(2)
        return stack
