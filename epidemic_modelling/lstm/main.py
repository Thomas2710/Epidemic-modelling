from epidemic_modelling.lstm.dataset import TimeSeriesDataset
from epidemic_modelling.pso import LSTMConfig
from epidemic_modelling.lstm.model import LSTMModel
from pytorch_lightning import Trainer


def main():
    config = LSTMConfig()

    dataset, sird = TimeSeriesDataset.load_data(config)
    sequences, labels, sird_initial, sird_final = TimeSeriesDataset.preprocess_data(
        dataset, sird, input_length=config.IN_DAYS, target_length=config.OUT_DAYS
    )
    TimeSeriesDataset.split_data(sequences, labels, sird_initial, sird_final)

    model = LSTMModel()
    trainer = Trainer(max_epochs=200, log_every_n_steps=5)
    trainer.fit(model, model.train_dataloader())
    trainer.test(model, model.test_dataloader())

    # save the model
    trainer.save_checkpoint("lstm_model.ckpt")


if __name__ == "__main__":
    main()