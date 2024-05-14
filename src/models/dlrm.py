import torch
import torch.nn as nn


class FeatureInteraction(nn.Module):
    def __init__(self, self_interaction: bool = True):
        super().__init__()
        self.self_interaction = self_interaction

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_dim = inputs.shape[1]

        concat_features = inputs.view(-1, feature_dim, 1)
        dot_products = torch.matmul(concat_features, concat_features.transpose(1, 2))
        ones = torch.ones_like(dot_products)

        mask = torch.triu(ones)
        out_dim = feature_dim * (feature_dim + 1) // 2

        flat_result = dot_products[mask.bool()]
        reshape_result = flat_result.view(-1, out_dim)

        return reshape_result


class DLRM(torch.nn.Module):
    def __init__(
        self,
        sparse_feature_number: int,
        dense_feature_number: int,
        num_embeddings: int,
        embed_dim: int,
        bottom_mlp_dims: list[int],
        top_mlp_dims: list[int],
        self_interaction: bool = True,
    ):
        super(DLRM, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embed_dim)
        self.layer_feature_interaction = FeatureInteraction(self_interaction)
        self.bottom_mlp = torch.nn.Sequential(
            torch.nn.Linear(dense_feature_number, bottom_mlp_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(bottom_mlp_dims[0], bottom_mlp_dims[1]),
            torch.nn.ReLU(),
        )
        self.top_mlp = torch.nn.Sequential(
            torch.nn.Linear(3278208, top_mlp_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(top_mlp_dims[0], top_mlp_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(top_mlp_dims[1], 1),
        )

    def forward(self, x_sparse: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:

        embed_x = self.embedding(x_sparse)
        embed_x = embed_x.view(x_sparse.shape[0], -1)

        bottom_mlp_output = self.bottom_mlp(x_dense)
        concat_first = torch.cat([bottom_mlp_output, embed_x], dim=-1)
        interaction = self.layer_feature_interaction(concat_first)

        concat_second = torch.cat([interaction, bottom_mlp_output], dim=-1)
        output = self.top_mlp(concat_second)
        output = output.squeeze().unsqueeze(1)

        return output
