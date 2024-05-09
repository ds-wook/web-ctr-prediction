from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

from generator import FeatureEngineering


class DataStorage:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def load_train_dataset(self) -> pd.DataFrame:
        train = pd.read_parquet(Path(self.cfg.data.path) / "train_sample.parquet")
        train = FeatureEngineering(self.cfg).generate_features(train)
        train_x = train.drop(columns=[*self.cfg.generator.drop_features, self.cfg.data.target])
        train_y = train[self.cfg.data.target]

        return train_x, train_y

    def load_test_dataset(self) -> pd.DataFrame:
        test = pd.read_parquet(Path(self.cfg.data.path) / "test.parquet")
        test = FeatureEngineering(self.cfg).generate_features(test)
        test_x = test.drop(columns=self.cfg.generator.drop_features)

        return test_x


def sampling_train_dataset(cfg: DictConfig) -> pd.DataFrame:
    pfile = pq.ParquetFile(Path(cfg.data.path) / "train.parquet")

    train = pd.DataFrame()
    chunksize = 10**7

    for chunk in tqdm(pfile.iter_batches(batch_size=chunksize), desc="Sampling data", leave=False):
        chunk = chunk.to_pandas()
        positive_sample = chunk[chunk["Click"] == 1]
        negative_sample = chunk[chunk["Click"] == 0].sample(frac=cfg.data.sampling, random_state=cfg.data.seed)
        train = pd.concat([train, negative_sample, positive_sample], axis=0, ignore_index=True)

    return train


def negative_sampling_train_dataset(cfg: DictConfig) -> pd.DataFrame:
    pfile = pq.ParquetFile(Path(cfg.data.path) / "train.parquet")

    train = pd.DataFrame()
    chunksize = 10**7

    for chunk in tqdm(pfile.iter_batches(batch_size=chunksize), desc="Sampling data", leave=False):
        chunk = chunk.to_pandas()
        # chunk = chunk.sample(frac=cfg.data.sampling, replace=False, random_state=602)
        neg_samp = chunk[chunk["Click"] == 0].sample(n=len(chunk[chunk["Click"] == 1]), random_state=cfg.data.seed)
        train = pd.concat([train, neg_samp, chunk[chunk["Click"] == 1]], axis=0)

    return train
