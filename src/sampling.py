from __future__ import annotations

import gc
from pathlib import Path

import hydra
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm


def negative_sampling_train_dataset(cfg: DictConfig) -> pd.DataFrame:
    pfile = pq.ParquetFile(Path(cfg.data.path) / "train_add_day.parquet")

    train = pd.DataFrame()
    negative = pd.DataFrame()
    positive = pd.DataFrame()
    chunksize = 10**7

    for chunk in tqdm(pfile.iter_batches(batch_size=chunksize), desc="Sampling data", leave=False):
        chunk = chunk.to_pandas()
        positive_sample = chunk[chunk["Click"] == 1]
        negative_sample = chunk[chunk["Click"] == 0]
        negative = pd.concat([negative, negative_sample], axis=0, ignore_index=True)
        positive = pd.concat([positive, positive_sample], axis=0, ignore_index=True)

    negative_sample = negative.sample(frac=cfg.data.sampling, replace=False, random_state=cfg.data.seed)
    train = pd.concat([train, negative_sample, positive], axis=0, ignore_index=True)
    del negative, positive

    return train


@hydra.main(config_path="../config/", config_name="sampling", version_base="1.2.0")
def _main(cfg: DictConfig):
    # load dataset
    train = negative_sampling_train_dataset(cfg)

    # save dataset
    train.to_parquet(Path(cfg.data.path) / f"{cfg.data.train}.parquet")

    del train
    gc.collect()


if __name__ == "__main__":
    _main()
