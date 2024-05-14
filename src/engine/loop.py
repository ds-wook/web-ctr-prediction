from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from tqdm import tqdm

from data.dataset import make_batchdata
from models import NeuMF
from utils import evaluation, update_avg


def train_step(
    cfg: DictConfig,
    model: NeuMF,
    optimizer: Optimizer,
    criterion: torch.nn.BCEWithLogitsLoss,
    ui_dataset: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Train epoch.
    Args:
        cfg: config
        model: model
        optimizer: optimizer
        criterion: criterion
        ui_dataset: ui dataset
    Returns:
        loss_dict: loss dict
    """
    model.train()
    curr_loss_avg = 0.0

    user_indices = np.arange(cfg.data.n_users)
    np.random.RandomState(cfg.models.epoch).shuffle(user_indices)
    batch_num = int(len(user_indices) / cfg.models.batch_size) + 1
    bar = tqdm(range(batch_num), leave=False)

    for step, batch_idx in enumerate(bar):
        user_ids, item_ids, feat0, feat1, feat2, feat3, labels = make_batchdata(
            user_indices, batch_idx, cfg.models.batch_size, ui_dataset
        )
        # 배치 사용자 단위로 학습
        user_ids = torch.LongTensor(user_ids).to(cfg.models.device)
        item_ids = torch.LongTensor(item_ids).to(cfg.models.device)
        feat0 = torch.FloatTensor(feat0).to(cfg.models.device)
        feat1 = torch.LongTensor(feat1).to(cfg.models.device)
        feat2 = torch.LongTensor(feat2).to(cfg.models.device)
        feat3 = torch.LongTensor(feat3).to(cfg.models.device)
        labels = torch.FloatTensor(labels).to(cfg.models.device)
        labels = labels.view(-1, 1)

        # grad 초기화
        optimizer.zero_grad()

        # 모델 forward
        output = model.forward(user_ids, item_ids, [feat0, feat1, feat2, feat3])
        output = output.view(-1, 1)

        loss = criterion(output, labels)

        # 역전파
        loss.backward()

        # 최적화
        optimizer.step()

        if torch.isnan(loss):
            print("Loss NAN. Train finish.")
            break

        curr_loss_avg = update_avg(curr_loss_avg, loss, step)

        msg = f"epoch: {cfg.models.epoch}, "
        msg += f"loss: {curr_loss_avg.item():.5f}, "
        msg += f"lr: {optimizer.param_groups[0]['lr']:.6f}"
        bar.set_description(msg)

    rets = {"losses": np.around(curr_loss_avg.item(), 5)}

    return rets


def valid_step(
    cfg: DictConfig,
    model: NeuMF,
    item_features: dict[str, dict[int, int]],
    user_features: dict[str, dict[int, int]],
    data: pd.DataFrame,
    mode: str = "valid",
) -> pd.DataFrame:
    """모델 평가
    Args:
        cfg: config
        model: 모델
        item_features: 아이템 피처
        user_features: 유저 피처
        data: 평가 데이터
        mode: 평가 모드
    Returns:
        sub: 추론 데이터
    """
    pred_list = []
    model.eval()
    meta_df = pd.read_csv(Path(cfg.data.path) / "meta_data.csv")

    # 추론할 모든 user array 집합
    query_user_ids = data["profile_id"].unique()

    # 추론할 모든 item array 집합
    full_item_ids = np.array([c for c in range(cfg.data.n_items)])

    full_item_ids_feat1 = [item_features["genre_mid"][c] for c in full_item_ids]
    for user_id in tqdm(query_user_ids, leave=False):
        with torch.no_grad():
            user_ids = np.full(cfg.data.n_items, user_id)

            user_ids = torch.LongTensor(user_ids).to(cfg.models.device)
            item_ids = torch.LongTensor(full_item_ids).to(cfg.models.device)

            feat0 = np.full(cfg.data.n_items, user_features["age"][user_id])
            feat0 = torch.FloatTensor(feat0).to(cfg.models.device)
            feat1 = torch.LongTensor(full_item_ids_feat1).to(cfg.models.device)
            feat2 = np.full(cfg.data.n_items, user_features["pr_interest_keyword_cd_1"][user_id])
            feat2 = torch.LongTensor(feat2).to(cfg.models.device)
            feat3 = np.full(cfg.data.n_items, user_features["ch_interest_keyword_cd_1"][user_id])
            feat3 = torch.LongTensor(feat3).to(cfg.models.device)
            eval_output = model.forward(user_ids, item_ids, [feat0, feat1, feat2, feat3]).detach().cpu().numpy()
            pred_u_score = eval_output.reshape(-1)

        pred_u_idx = np.argsort(pred_u_score)[::-1]
        pred_u = full_item_ids[pred_u_idx]
        pred_list.append(list(pred_u[: cfg.models.top_k]))

    pred = pd.DataFrame()
    pred["profile_id"] = query_user_ids
    pred["predicted_list"] = pred_list

    return evaluation(data, pred, meta_df, cfg), pred if mode == "valid" else pred
