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

    def forward(self, input_seq, sird_info):
        # TODO:
        # load params beta, gamma,delta
        # RUN SIRD FOR 1 WEEK
        # Compare computed S,I,R,D values with the real ones

        # shape of input_seq: (batch_size, seq_len, feature_size)

        lstm_out, _ = self.lstm(input_seq)
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        self.sird.setup(predictions[:, 0], predictions[:, 1], predictions[:, 2])
        sird_info = sird_info
        sol = odeint(
            self.sird, sird_info, torch.linspace(0, self.offset, self.offset + 1)
        )
        # TODO: return all solutions
        return sol[-1]

        # runno il sird con questi parametri
        # calcolo la loss di S I R D?

        # print(predictions.shape)
        # return predictions.unsqueeze(1)

    def training_step(self, batch, batch_idx):
        sequences, _, sird_initial, sird_final = batch
        y_pred = self(sequences, sird_initial.float())
        # TODO: check on what are you using the loss -> all S,I,R,D computed vals vs real ones
        loss = self.criterion(y_pred[1:], sird_final.float()[1:])
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        sequences, labels, sird_initial, sird_final = batch
        y_pred = self(sequences, sird_initial.float())
        loss = self.criterion(y_pred[1:], sird_final.float()[1:])
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
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, t, y):
        return self._get_eqns(y)

    def _get_eqns(self, y):
        # shape is [8,1,4], i want to split it into 4 tensors (innermost dimension)
        S, I, R, D = torch.split(y, 1, dim=2)
        S = S.squeeze()
        I = I.squeeze()
        R = R.squeeze()
        D = D.squeeze()
        N = self.population
        dSdt = -self.beta * S * I / N
        dIdt = (self.beta * S * I - self.gamma * I - self.delta * I) / N
        dRdt = self.gamma * I / N
        dDdt = self.delta * I / N
        # convert [8], [8], [8], [8] to [8,1,4]
        stack = torch.stack([dSdt, dIdt, dRdt, dDdt], dim=1).unsqueeze(1)
        return stack
