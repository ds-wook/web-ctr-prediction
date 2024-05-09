from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from omegaconf import DictConfig
from tqdm import tqdm

from data import DataStorage
from models import BulidModel


def inference_models(models: list[BulidModel], test_x: pd.DataFrame) -> np.ndarray:
    """
    Given a model, predict probabilities for each class.
    Args:
        models: Models
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """

    folds = len(models)
    preds = []

    for model in tqdm(models, total=folds, desc="Predicting models"):
        if isinstance(model, xgb.Booster):
            preds.append(model.predict(xgb.DMatrix(test_x)))

        elif isinstance(model, lgb.Booster):
            preds.append(model.predict(test_x))

        elif isinstance(model, CatBoostClassifier):
            preds.append(model.predict_proba(test_x)[:, 1])

        else:
            raise ValueError("Model not supported")

    return np.mean(preds, axis=0)


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    # load model
    result = joblib.load(Path(cfg.models.path) / f"{cfg.models.results}.pkl")

    # load dataset
    data_storage = DataStorage(cfg)
    test = pd.read_parquet(Path(cfg.data.path) / f"{cfg.data.test}.parquet")
    test_x = data_storage.load_test_dataset(test)

    # predict
    submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")
    submit[cfg.data.target] = inference_models(result, test_x)
    submit.to_csv(Path(cfg.output.path) / f"{cfg.output.name}.csv", index=False)


if __name__ == "__main__":
    _main()
