from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from deepctr_torch.models import DeepFM
from omegaconf import DictConfig
from tqdm import tqdm

from data import DataStorage
from models import ModelResult


def inference_models(results: ModelResult, test_x: pd.DataFrame | dict[str, pd.Series]) -> np.ndarray:
    """Given a model, predict probabilities for each class.
    Args:
        results: ModelResult object
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """

    folds = len(results.models)
    preds = np.zeros((test_x.shape[0],))

    for model in tqdm(results.models.values(), total=folds, desc="Predicting models"):
        if isinstance(model, lgb.Booster):
            preds += model.predict(test_x) / folds

        elif isinstance(model, xgb.Booster):
            preds += model.predict(xgb.DMatrix(test_x)) / folds

        elif isinstance(model, DeepFM):
            preds += model.predict(test_x, 64).flatten() / folds

        else:
            preds += model.predict_proba(test_x)[:, 1] / folds

    return preds


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    # load model
    result = joblib.load(Path(cfg.models.path) / f"{cfg.models.results}.pkl")

    # load test dataset
    data_storage = DataStorage(cfg)
    test_x = data_storage.load_test_dataset()

    test_x = {name: test_x[name] for name in test_x.columns} if cfg.models.name == "deepfm" else test_x

    # load submit dataset
    submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")

    # predict
    preds = inference_models(result, test_x)
    submit[cfg.data.target] = preds
    submit.to_csv(Path(cfg.output.path) / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
