import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=20):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class ActionEncoder(nn.Module):
    def __init__(
        self,
        horizon=32,
        input_dim=2,
        hidden_dims=None,
        latent_dim=128,
        use_velocity: bool = False,
    ):
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.planning_horizon = horizon  # Why am I assigning this :/
        self.use_velocity = use_velocity

        actual_input_dim = input_dim + 1 if self.use_velocity else input_dim

        self.input_projection = nn.Linear(actual_input_dim, hidden_dims[0])

        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dims[0],
            max_seq_length=horizon,
        )

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

        self.final_layer = nn.Linear(hidden_dims[-1], latent_dim)

        # self.global_avg_pool = lambda x: torch.mean(x, dim=1)

    def forward(self, x, velocity=None):
        if velocity is not None:
            velocity_expanded = velocity.unsqueeze(1).expand(-1, x.size(1), 1)
            x = torch.cat([x, velocity_expanded], dim=2)

        x = self.input_projection(x)

        # x = self.positional_encoding(x)

        batch_size, seq_length, _ = x.shape
        x = x.reshape(batch_size * seq_length, -1)
        x = self.mlp(x)
        x = x.reshape(batch_size, seq_length, -1)

        x = self.final_layer(x)

        # encoded = self.global_avg_pool(x)

        return x
