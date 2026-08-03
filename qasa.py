"""Gated QASA model for the baseline training and evaluation paths."""

import math
import random

import torch
import torch.nn as nn

from qasa_mlp import MlpDecoder
from qasa_transformer import TransformerDecoder
from slot_attn import SlotAttentionEncoder
from qasa_utils import linear, spiral_pattern


class QASA(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.which_encoder = args.which_encoder
        self.encoder = encoder
        self.encoder_final_norm = args.encoder_final_norm

        for name, parameter in self.encoder.named_parameters():
            if "blocks" in name:
                block_id = int(name.split(".")[1])
                parameter.requires_grad = block_id >= args.finetune_blocks_after
            else:
                parameter.requires_grad = False

        with torch.no_grad():
            device = next(self.encoder.parameters()).device
            image = torch.rand(
                1, args.img_channels, args.image_size, args.image_size, device=device
            )
            encoded = self.forward_encoder(image)
            _, num_tokens, d_model = encoded.shape

        args.d_model = d_model
        self.num_slots = args.num_slots
        self.d_model = d_model
        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations,
            args.num_slots,
            d_model,
            args.slot_size,
            args.mlp_hidden_size,
            args.pos_channels,
            args.truncate,
            args.init_method,
        )

        self.dec_type = args.dec_type
        self.slot_proj = nn.Sequential(
            linear(args.slot_size, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

        if self.dec_type == "transformer":
            self.input_proj = nn.Sequential(
                linear(d_model, d_model, bias=False),
                nn.LayerNorm(d_model),
            )
            self.dec = TransformerDecoder(
                args.num_dec_blocks,
                args.max_tokens,
                d_model,
                args.num_heads,
                args.dropout,
                args.num_cross_heads,
            )
            self._init_permutations(num_tokens, args.train_permutations, args.eval_permutations)
            self.bos_tokens = nn.Parameter(
                torch.zeros(len(self.permutations), 1, 1, d_model)
            )
            nn.init.normal_(self.bos_tokens, std=0.02)

            self.dec_slots_attns = []

            def capture_cross_attention(module, inputs):
                self.dec_slots_attns = [inputs[0].detach().contiguous().clone()]

            attention_dropout = self.dec.blocks[-1].encoder_decoder_attn.attn_dropout
            self.remove_handle = attention_dropout.register_forward_pre_hook(
                capture_cross_attention
            )
        elif self.dec_type == "mlp":
            self.dec = MlpDecoder(
                d_model, d_model, args.max_tokens, args.mlp_dec_hidden
            )
            self.register_buffer("bos_tokens", None)
        else:
            raise ValueError(f"Unknown decoder type: {self.dec_type}")

        self.use_conditional_slot_pruning = args.use_conditional_slot_pruning
        self.cov_rho = float(args.cov_rho)
        self.cov_tau = float(args.cov_tau)
        self.cov_kmin = int(args.cov_kmin)
        self.gate_eps = float(args.gate_eps)
        self.gate_layers = args.gate_layers
        self.cov_novelty_alpha = args.cov_novelty_alpha

    def _init_permutations(self, num_tokens, train_permutations, eval_permutations):
        size = math.isqrt(num_tokens)
        if size * size != num_tokens:
            raise ValueError("The encoder token count must be a perfect square")

        standard = torch.arange(num_tokens)
        self.train_permutations = train_permutations
        if train_permutations == "standard":
            self.permutations = [standard]
            self.eval_permutations = "standard"
        else:
            grid = standard.reshape(size, size)
            top_left = torch.tensor(
                [grid[row, col] for col in range(size) for row in range(size)]
            )
            top_right = torch.tensor(
                [grid[row, col] for col in range(size - 1, -1, -1) for row in range(size)]
            )
            right_top = torch.tensor(
                [grid[row, col] for row in range(size) for col in range(size - 1, -1, -1)]
            )
            bottom_right = torch.tensor(
                [
                    grid[row, col]
                    for col in range(size - 1, -1, -1)
                    for row in range(size - 1, -1, -1)
                ]
            )
            right_bottom = torch.tensor(
                [
                    grid[row, col]
                    for row in range(size - 1, -1, -1)
                    for col in range(size - 1, -1, -1)
                ]
            )
            bottom_left = torch.tensor(
                [grid[row, col] for col in range(size) for row in range(size - 1, -1, -1)]
            )
            left_bottom = torch.tensor(
                [grid[row, col] for row in range(size - 1, -1, -1) for col in range(size)]
            )
            spiral = torch.tensor(
                spiral_pattern(grid, how="top_right")[::-1].copy()
            )
            self.permutations = [
                standard,
                top_left,
                top_right,
                right_top,
                bottom_right,
                right_bottom,
                bottom_left,
                left_bottom,
                spiral,
            ]
            self.eval_permutations = eval_permutations
        self.perm_ind = list(range(len(self.permutations)))

    def _build_slot_gate(self, slots_attns):
        with torch.no_grad():
            attention = slots_attns.detach()
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6)
            batch_size, num_tokens, num_slots = attention.shape

            winners = attention.argmax(dim=-1)
            winner_weights = attention.gather(-1, winners.unsqueeze(-1)).squeeze(-1)
            winner_mass = torch.zeros(
                batch_size,
                num_slots,
                device=attention.device,
                dtype=attention.dtype,
            )
            winner_mass.scatter_add_(1, winners, winner_weights)
            quality = winner_mass / (attention.sum(dim=1) + 1e-6)
            gate = torch.full_like(quality, self.gate_eps)

            for batch_index in range(batch_size):
                sample_attention = attention[batch_index]
                order = torch.argsort(quality[batch_index], descending=True)
                active = torch.zeros(
                    num_slots, dtype=torch.bool, device=attention.device
                )
                keep = max(self.cov_kmin, 1)
                active[order[:keep]] = True

                def coverage(mask):
                    if mask.any():
                        covered_mass = sample_attention[:, mask].sum(dim=1)
                    else:
                        covered_mass = torch.zeros(
                            num_tokens,
                            device=attention.device,
                            dtype=attention.dtype,
                        )
                    return covered_mass >= self.cov_tau

                covered = coverage(active)
                covered_fraction = covered.float().mean().item()
                index = keep
                while covered_fraction < self.cov_rho and index < num_slots:
                    slot_index = int(order[index])
                    index += 1
                    if active[slot_index]:
                        continue

                    if self.cov_novelty_alpha is not None:
                        total_mass = sample_attention[:, slot_index].sum()
                        if total_mass <= 1e-6:
                            continue
                        covered_mass = sample_attention[covered, slot_index].sum()
                        novelty = 1.0 - (covered_mass / (total_mass + 1e-6)).item()
                        if novelty < float(self.cov_novelty_alpha):
                            continue

                    active[slot_index] = True
                    covered = coverage(active)
                    covered_fraction = covered.float().mean().item()

                gate[batch_index, active] = 1.0
            return gate

    def forward_encoder(self, image):
        self.encoder.eval()
        if self.which_encoder == "dinov2_vits14":
            encoded = self.encoder.prepare_tokens_with_masks(image, None)
        elif self.which_encoder == "dino_vitb16":
            encoded = self.encoder.prepare_tokens(image)
        else:
            raise ValueError(f"Unsupported encoder: {self.which_encoder}")

        for block in self.encoder.blocks:
            encoded = block(encoded)
        if self.encoder_final_norm:
            encoded = self.encoder.norm(encoded)
        return encoded[:, 1:]

    def _selected_permutations(self):
        mode = self.train_permutations if self.training else self.eval_permutations
        if mode == "standard":
            return [0]
        if mode == "random":
            return [random.choice(self.perm_ind)]
        if mode == "all":
            return self.perm_ind
        raise ValueError(f"Unknown permutation mode: {mode}")

    def forward_decoder(self, slots, target, gate=None):
        decoder_slots = self.slot_proj(slots)
        if self.dec_type == "mlp":
            reconstruction, decoder_attention = self.dec(decoder_slots, gate=gate)
            return reconstruction, decoder_attention.transpose(1, 2)

        all_attention = []
        all_outputs = []
        for permutation_id in self._selected_permutations():
            permutation = self.permutations[permutation_id]
            bos = self.bos_tokens[permutation_id].expand(target.shape[0], -1, -1)
            decoder_input = torch.cat(
                (bos, target[:, permutation, :][:, :-1, :]), dim=1
            )
            decoder_input = self.input_proj(decoder_input)
            decoder_output = self.dec(
                decoder_input,
                decoder_slots,
                causal_mask=True,
                gate=gate,
                gate_layers=self.gate_layers,
            )

            decoder_attention = self.dec_slots_attns[0]
            self.dec_slots_attns = []
            decoder_attention = decoder_attention.sum(dim=1)
            decoder_attention = decoder_attention / decoder_attention.sum(
                dim=2, keepdim=True
            )
            inverse = torch.argsort(permutation)
            all_attention.append(decoder_attention[:, inverse, :])
            all_outputs.append(decoder_output[:, inverse, :])

        return (
            torch.stack(all_outputs).mean(0),
            torch.stack(all_attention).mean(0),
        )

    def forward(self, image, gate_wp=False, skip_norm=False):
        batch_size = image.shape[0]
        encoded = self.forward_encoder(image)
        target = encoded.detach().clone()
        slots, slots_attns, _, attn_logits = self.slot_attn(encoded, skip_norm)

        gate = None
        if self.use_conditional_slot_pruning and not gate_wp:
            gate = self._build_slot_gate(slots_attns)

        reconstruction, decoder_attention = self.forward_decoder(
            slots, target, gate=gate
        )
        loss = torch.mean((target - reconstruction) ** 2)

        spatial_size = math.isqrt(target.shape[1])
        slots_attns = slots_attns.transpose(-1, -2).reshape(
            batch_size, self.num_slots, spatial_size, spatial_size
        )
        decoder_attention = decoder_attention.transpose(-1, -2).reshape(
            batch_size, self.num_slots, spatial_size, spatial_size
        )
        return (
            loss,
            slots_attns,
            decoder_attention,
            slots,
            reconstruction,
            attn_logits.squeeze(),
            gate,
        )

    def load_compatible_state_dict(self, state_dict, strict=True):
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            normalized[key.replace("tf_dec.", "dec.")] = value

        current = self.state_dict()
        skipped_masks = set()
        for key in list(normalized):
            if key.startswith("second_encoder.") and key not in current:
                normalized.pop(key)
            elif (
                key in current
                and normalized[key].shape != current[key].shape
                and "dec.blocks" in key
                and key.endswith("self_attn_mask")
            ):
                normalized.pop(key)
                skipped_masks.add(key)

        result = super().load_state_dict(normalized, strict=False)
        if strict:
            missing = [key for key in result.missing_keys if key not in skipped_masks]
            unexpected = list(result.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
                )
        return result
