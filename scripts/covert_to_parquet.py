from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    train = pd.DataFrame()

    for chunk in tqdm(pd.read_csv(Path(cfg.data.path) / "train.csv", chunksize=1000000)):
        train = pd.concat([train, chunk])

    train.to_parquet(Path(cfg.data.path) / "train.parquet")

    test = pd.read_csv(Path(cfg.data.path) / "test.csv")
    test.to_parquet(Path(cfg.data.path) / "test.parquet")


if __name__ == "__main__":
    _main()
