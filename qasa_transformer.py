"""Transformer decoder used by the gated QASA path.

The module names intentionally match the original implementation so existing
checkpoints keep the same state_dict keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from qasa_utils import linear


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, gain=1.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)

    def forward(self, q, k, v, attn_mask=None, gate=None):
        batch_size, target_len, _ = q.shape
        source_len = k.shape[1]

        q = self.proj_q(q).view(batch_size, target_len, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(batch_size, source_len, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(batch_size, source_len, self.num_heads, -1).transpose(1, 2)
        q = q * (q.shape[-1] ** -0.5)

        if gate is not None:
            projected_gate = gate.clamp_min(1e-9).view(batch_size, 1, source_len, 1)
            k = k * projected_gate
            v = v * projected_gate

        logits = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask, float("-inf"))
        if gate is not None:
            logits = logits + torch.log(gate.clamp_min(1e-6)).view(
                batch_size, 1, 1, source_len
            )

        attn = self.attn_dropout(F.softmax(logits, dim=-1))
        output = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, target_len, -1)
        return self.output_dropout(self.proj_o(output))


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        max_len,
        d_model,
        num_heads,
        dropout=0.0,
        gain=1.0,
        is_first=False,
        num_cross_heads=None,
    ):
        super().__init__()
        self.is_first = is_first
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        self.self_attn_mask = nn.Parameter(mask, requires_grad=False)
        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attn = MultiHeadAttention(
            d_model, num_cross_heads or num_heads, dropout, gain
        )
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init="kaiming"),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout),
        )

    def forward(self, inputs, encoder_output, causal_mask=True, gate=None):
        target_len = inputs.shape[1]
        self_attn_mask = self.self_attn_mask[:target_len, :target_len] if causal_mask else None

        if self.is_first:
            inputs = self.self_attn_layer_norm(inputs)
            inputs = inputs + self.self_attn(inputs, inputs, inputs, self_attn_mask)
        else:
            normalized = self.self_attn_layer_norm(inputs)
            inputs = inputs + self.self_attn(normalized, normalized, normalized, self_attn_mask)

        normalized = self.encoder_decoder_attn_layer_norm(inputs)
        inputs = inputs + self.encoder_decoder_attn(
            normalized, encoder_output, encoder_output, gate=gate
        )
        return inputs + self.ffn(self.ffn_layer_norm(inputs))


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_blocks,
        max_len,
        d_model,
        num_heads,
        dropout=0.0,
        num_cross_heads=None,
    ):
        super().__init__()
        if num_blocks > 0:
            gain = (3 * num_blocks) ** -0.5
            self.blocks = nn.ModuleList(
                [
                    TransformerDecoderBlock(
                        max_len,
                        d_model,
                        num_heads,
                        dropout,
                        gain,
                        is_first=index == 0,
                        num_cross_heads=num_cross_heads,
                    )
                    for index in range(num_blocks)
                ]
            )
        else:
            self.blocks = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        inputs,
        encoder_output,
        causal_mask=True,
        gate=None,
        gate_layers="all",
    ):
        last_index = len(self.blocks) - 1
        for index, block in enumerate(self.blocks):
            use_gate = gate is not None and (gate_layers == "all" or index == last_index)
            inputs = block(
                inputs,
                encoder_output,
                causal_mask,
                gate=gate if use_gate else None,
            )
        return self.layer_norm(inputs)
