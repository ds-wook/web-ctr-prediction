from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


def file_df_to_parquet(local_file: Path) -> None:
    for i, chunk in enumerate(pd.read_csv(local_file / "train.csv", chunksize=1000000)):
        chunk.to_parquet(local_file / f"train_sample_{i}.parquet")


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    file_df_to_parquet(Path(cfg.data.path))
    train = pd.read_parquet(Path(cfg.data.path) / "train_sample_0.parquet")

    for i in tqdm(range(1, 29)):
        train = pd.concat([train, pd.read_parquet(Path(cfg.data.path) / f"train_sample_{i}.parquet")])

    train.to_parquet(Path(cfg.data.path) / "train.parquet")


if __name__ == "__main__":
    _main()
