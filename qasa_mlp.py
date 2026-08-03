"""MLP decoder used by the gated QASA path."""

import torch
from torch import nn


class MlpDecoder(nn.Module):
    def __init__(self, object_dim, output_dim, num_patches, hidden_features=2048):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, object_dim) * 0.02)
        self.decoder = build_mlp(object_dim, output_dim + 1, hidden_features)

    def forward(self, encoder_output, gate=None):
        batch_size, num_slots, _ = encoder_output.shape
        slots = encoder_output.flatten(0, 1)
        slots = slots.unsqueeze(1).expand(-1, self.num_patches, -1)
        decoded = self.decoder(slots + self.pos_embed)
        decoded = decoded.unflatten(0, (batch_size, num_slots))
        decoded_patches, alpha_logits = decoded.split([self.output_dim, 1], dim=-1)

        alpha_logits = alpha_logits.squeeze(-1)
        if gate is not None:
            inactive = gate <= 0.5
            alpha_logits = alpha_logits.masked_fill(inactive.unsqueeze(-1), -1e9)
        alpha = alpha_logits.softmax(dim=1).unsqueeze(-1)

        reconstruction = torch.sum(decoded_patches * alpha, dim=1)
        masks = alpha.squeeze(-1)
        return reconstruction, masks


def build_mlp(input_dim, output_dim, hidden_features=2048, n_hidden_layers=3):
    layers = []
    current_dim = input_dim
    for _ in range(n_hidden_layers):
        layer = nn.Linear(current_dim, hidden_features)
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.ReLU(inplace=True)])
        current_dim = hidden_features

    output_layer = nn.Linear(current_dim, output_dim)
    nn.init.zeros_(output_layer.bias)
    layers.append(output_layer)
    return nn.Sequential(*layers)
