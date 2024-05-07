from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig

from data import DataStorage
from models import bulid_model


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        data_storage = DataStorage(cfg)

        # load dataset
        train_x, train_y = data_storage.load_train_data()

        # choose trainer
        trainer = bulid_model(cfg)

        # train model
        trainer.run_cv_training(train_x, train_y)

        # save model
        trainer.save_model(Path(cfg.models.path))


if __name__ == "__main__":
    _main()
