from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
from catboost import CatBoostClassifier
from deepctr_torch.models import WDL, AutoInt, FiBiNET, xDeepFM
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing_extensions import Self


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]


class BaseModel(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        raise NotImplementedError

    def save_model(self, save_dir: Path) -> None:
        joblib.dump(self.result, save_dir / f"{self.cfg.models.results}.pkl")

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> Any:
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def _predict(self, model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(model, xgb.Booster):
            return model.predict(xgb.DMatrix(X))

        elif isinstance(model, CatBoostClassifier):
            return model.predict_proba(X)[:, 1]

        elif isinstance(model, WDL | xDeepFM | AutoInt | FiBiNET):
            feature_names = [*self.cfg.generator.sparse_features, *self.cfg.generator.dense_features]
            valid_model_input = {name: X[name] for name in feature_names}

            return model.predict(valid_model_input, batch_size=512).flatten()

        elif isinstance(model, lgb.Booster):
            return model.predict(X)

        else:
            raise ValueError("Model not supported")

    def run_cv_training(self, X: pd.DataFrame, y: pd.Series) -> Self:
        oof_preds = np.zeros(X.shape[0])
        models = {}
        kfold = StratifiedKFold(n_splits=self.cfg.data.n_splits, shuffle=True, random_state=self.cfg.data.seed)

        for fold, (train_idx, valid_idx) in enumerate(iterable=kfold.split(X, y), start=1):
            with wandb.init(project="competition", name=f"{self.cfg.models.name}-fold-{fold}", dir="never"):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = self.fit(X_train, y_train, X_valid, y_valid)
                oof_preds[valid_idx] = self._predict(model, X_valid)

                models[f"fold_{fold}"] = model

        del model, X_train, X_valid, y_train, y_valid
        gc.collect()

        self.result = ModelResult(oof_preds=oof_preds, models=models)

        print(f"CV Score: {roc_auc_score(y, oof_preds):.6f}")

        del oof_preds, y, models
        gc.collect()

        return self
