from omegaconf import DictConfig

from .base import *
from .boosting import *
from .boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer
from .dlrm import AutoIntTrainer, FiBiNetTranier, WDLTrainer, XDeepFMTrainer

BulidModel = (
    CatBoostTrainer
    | LightGBMTrainer
    | XGBoostTrainer
    | WDLTrainer
    | FiBiNetTranier
    | XDeepFMTrainer
    | AutoIntTrainer
)


def build_model(cfg: DictConfig) -> BulidModel:
    model_type = {
        "lightgbm": LightGBMTrainer(cfg),
        "xgboost": XGBoostTrainer(cfg),
        "catboost": CatBoostTrainer(cfg),
        "wdl": WDLTrainer(cfg),
        "fibinet": FiBiNetTranier(cfg),
        "xdeepfm": XDeepFMTrainer(cfg),
        "autoint": AutoIntTrainer(cfg),
    }

    if trainer := model_type.get(cfg.models.name):
        return trainer

    else:
        raise NotImplementedError(f"Model '{cfg.models.name}' is not implemented.")
