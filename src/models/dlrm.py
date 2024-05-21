import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from deepctr_torch.callbacks import EarlyStopping
from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from deepctr_torch.models import WDL, DeepFM
from omegaconf import DictConfig
from pytorch_optimizer import MADGRAD

from models import BaseModel


class DeepFMTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> DeepFM:

        feature_columns = [
            SparseFeat(feat, vocabulary_size=X_train[feat].nunique(), embedding_dim=16)
            for feat in self.cfg.generator.sparse_features
        ] + [DenseFeat(feat, 1) for feat in self.cfg.generator.dense_features]

        feature_names = get_feature_names(feature_columns)

        train_model_input = {name: X_train[name] for name in feature_names}
        valid_model_input = {name: X_valid[name] for name in feature_names}

        model = DeepFM(
            dnn_feature_columns=feature_columns,
            linear_feature_columns=feature_columns,
            device=self.cfg.models.device,
            seed=self.cfg.models.seed,
            l2_reg_linear=self.cfg.models.l2_reg_linear,
            l2_reg_embedding=self.cfg.models.l2_reg_embedding,
            dnn_activation=self.cfg.models.dnn_activation,
            dnn_dropout=self.cfg.models.dnn_dropout,
            dnn_use_bn=True,
        )

        model.compile(
            MADGRAD(model.parameters(), lr=self.cfg.models.lr),
            "binary_crossentropy",
            metrics=["binary_crossentropy", "auc"],
        )

        es = EarlyStopping(
            monitor="val_binary_crossentropy",
            min_delta=0,
            verbose=self.cfg.models.verbose,
            patience=self.cfg.models.patience,
            mode=self.cfg.models.mode,
        )

        model.fit(
            train_model_input,
            y_train.values,
            batch_size=self.cfg.models.batch_size,
            epochs=self.cfg.models.epochs,
            verbose=self.cfg.models.verbose,
            validation_data=(valid_model_input, y_valid.values),
            callbacks=[es],
        )

        return model


class WDLTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> DeepFM:

        feature_columns = [
            SparseFeat(feat, vocabulary_size=X_train[feat].nunique(), embedding_dim=16)
            for feat in self.cfg.generator.sparse_features
        ] + [DenseFeat(feat, 1) for feat in self.cfg.generator.dense_features]

        feature_names = get_feature_names(feature_columns)

        train_model_input = {name: X_train[name] for name in feature_names}
        valid_model_input = {name: X_valid[name] for name in feature_names}

        model = WDL(
            dnn_feature_columns=feature_columns,
            linear_feature_columns=feature_columns,
            device=self.cfg.models.device,
            seed=self.cfg.models.seed,
            l2_reg_linear=self.cfg.models.l2_reg_linear,
            l2_reg_embedding=self.cfg.models.l2_reg_embedding,
            dnn_activation=self.cfg.models.dnn_activation,
            dnn_dropout=self.cfg.models.dnn_dropout,
            dnn_use_bn=True,
        )

        model.compile(
            MADGRAD(model.parameters(), lr=self.cfg.models.lr),
            "binary_crossentropy",
            metrics=["binary_crossentropy", "auc"],
        )

        es = EarlyStopping(
            monitor="val_binary_crossentropy",
            min_delta=0,
            verbose=self.cfg.models.verbose,
            patience=self.cfg.models.patience,
            mode=self.cfg.models.mode,
        )

        model.fit(
            train_model_input,
            y_train.values,
            batch_size=self.cfg.models.batch_size,
            epochs=self.cfg.models.epochs,
            verbose=self.cfg.models.verbose,
            validation_data=(valid_model_input, y_valid.values),
            callbacks=[es],
        )

        return model


class FeatureInteraction(nn.Module):
    def __init__(self, self_interaction: bool):
        super().__init()
        self.self_interaction = self_interaction

    def forward(self, inputs: torch.Tensor):
        feature_dim = inputs.shape[1]

        concat_features = inputs.view(-1, feature_dim, 1)
        dot_products = torch.matmul(concat_features, concat_features.transpose(1, 2))
        ones = torch.ones_like(dot_products)

        mask = torch.triu(ones)
        out_dim = feature_dim * (feature_dim + 1) // 2

        flat_result = dot_products[mask.bool()]
        reshape_result = flat_result.view(-1, out_dim)

        return reshape_result

