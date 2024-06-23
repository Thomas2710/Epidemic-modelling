from pytorch_lightning import Trainer

from epidemic_modelling.lstm.dataset import TimeSeriesDataset
from epidemic_modelling.lstm.model import LSTMModel


def train(config):

    dataset, sird = TimeSeriesDataset.load_data(config)
    starting_params, original_sird = TimeSeriesDataset.preprocess_data(
        dataset, sird, input_length=config.IN_WEEKS, target_length=config.OUT_WEEKS
    )
    TimeSeriesDataset.split_data(starting_params, original_sird)

    model = LSTMModel()
    trainer = Trainer(
        max_epochs=config.EPOCHS, log_every_n_steps=config.LOG_EVERY_N_STEPS
    )
    trainer.fit(model, model.train_dataloader())
    trainer.test(model, model.test_dataloader())

    # save the model
    trainer.save_checkpoint("lstm_model.ckpt")
