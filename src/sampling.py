from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from data import sampling_train


@hydra.main(config_path="../config/", config_name="sampling", version_base="1.2.0")
def _main(cfg: DictConfig):
    # load dataset
    train = sampling_train(cfg)

    # save dataset
    train.to_parquet(Path(cfg.data.path) / "train_sample.parquet")


if __name__ == "__main__":
    _main()
