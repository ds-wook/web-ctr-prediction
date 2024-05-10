from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from data import negative_sampling_train_dataset, sampling_train_dataset


@hydra.main(config_path="../config/", config_name="sampling", version_base="1.2.0")
def _main(cfg: DictConfig):
    # load dataset
    train = negative_sampling_train_dataset(cfg) if cfg.mode == "negative_sampling" else sampling_train_dataset(cfg)

    # save dataset
    train.to_parquet(Path(cfg.data.path) / f"{cfg.data.train}.parquet")


if __name__ == "__main__":
    _main()
