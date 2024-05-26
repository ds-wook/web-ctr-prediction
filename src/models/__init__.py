from omegaconf import DictConfig

from .base import *
from .boosting import *
from .boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer
from .dlrm import DeepFMTrainer, DIFMTrainer, WDLTrainer, XDeepFMTrainer

BulidModel = (
    CatBoostTrainer | LightGBMTrainer | XGBoostTrainer | DeepFMTrainer | WDLTrainer | DIFMTrainer | XDeepFMTrainer
)


def build_model(cfg: DictConfig) -> BulidModel:
    model_type = {
        "lightgbm": LightGBMTrainer(cfg),
        "xgboost": XGBoostTrainer(cfg),
        "catboost": CatBoostTrainer(cfg),
        "deepfm": DeepFMTrainer(cfg),
        "wdl": WDLTrainer(cfg),
        "difm": DIFMTrainer(cfg),
        "xdeepfm": XDeepFMTrainer(cfg),
    }

    if trainer := model_type.get(cfg.models.name):
        return trainer

    else:
        raise NotImplementedError(f"Model '{cfg.models.name}' is not implemented.")
