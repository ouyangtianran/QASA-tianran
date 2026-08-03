"""Frozen QASA slot extraction for downstream probes."""

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from checkpoints import extract_state_dict, load_checkpoint
from encoders import PATCH_SIZES, build_encoder
from slot_attn import SlotAttentionEncoder


ARCH_DEFAULTS: Dict[str, Any] = {
    "which_encoder": None,
    "image_size": 224,
    "encoder_final_norm": False,
    "num_iterations": 3,
    "num_slots": None,
    "slot_size": None,
    "mlp_hidden_size": 1024,
    "pos_channels": 4,
    "truncate": "none",
    "init_method": None,
    "skip_norm": False,
    "use_conditional_slot_pruning": True,
    "cov_rho": 0.8,
    "cov_tau": 0.5,
    "cov_kmin": 2,
    "cov_novelty_alpha": None,
}

ARCH_KEYS = tuple(ARCH_DEFAULTS)
ENCODER_DIMS = {
    "dino_vitb16": 768,
    "dinov2_vits14": 384,
}


def build_active_slot_mask(
    attention: torch.Tensor,
    cov_rho: float,
    cov_tau: float,
    cov_kmin: int,
    cov_novelty_alpha: Optional[float],
) -> torch.Tensor:
    """Apply the same quality-guided slot selection used by QASA."""
    with torch.no_grad():
        attention = attention.detach()
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
        active_mask = torch.zeros(
            batch_size,
            num_slots,
            device=attention.device,
            dtype=torch.bool,
        )

        for batch_index in range(batch_size):
            sample_attention = attention[batch_index]
            order = torch.argsort(quality[batch_index], descending=True)
            keep = min(max(int(cov_kmin), 1), num_slots)
            active_mask[batch_index, order[:keep]] = True

            def coverage(mask: torch.Tensor) -> torch.Tensor:
                covered_mass = sample_attention[:, mask].sum(dim=1)
                return covered_mass >= cov_tau

            covered = coverage(active_mask[batch_index])
            covered_fraction = covered.float().mean().item()
            index = keep
            while covered_fraction < cov_rho and index < num_slots:
                slot_index = int(order[index])
                index += 1

                if cov_novelty_alpha is not None:
                    total_mass = sample_attention[:, slot_index].sum()
                    if total_mass <= 1e-6:
                        continue
                    covered_mass = sample_attention[covered, slot_index].sum()
                    novelty = 1.0 - (covered_mass / (total_mass + 1e-6)).item()
                    if novelty < cov_novelty_alpha:
                        continue

                active_mask[batch_index, slot_index] = True
                covered = coverage(active_mask[batch_index])
                covered_fraction = covered.float().mean().item()

        return active_mask


class QASASlotExtractor(nn.Module):
    def __init__(self, encoder: nn.Module, slot_attn: nn.Module, config: SimpleNamespace):
        super().__init__()
        self.encoder = encoder
        self.slot_attn = slot_attn
        self.which_encoder = config.which_encoder
        self.encoder_final_norm = bool(config.encoder_final_norm)
        self.skip_norm = bool(config.skip_norm)
        self.use_conditional_slot_pruning = bool(config.use_conditional_slot_pruning)
        self.cov_rho = float(config.cov_rho)
        self.cov_tau = float(config.cov_tau)
        self.cov_kmin = int(config.cov_kmin)
        self.cov_novelty_alpha = config.cov_novelty_alpha

    def forward_encoder(self, image: torch.Tensor) -> torch.Tensor:
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

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.forward_encoder(image)
        slots, attention, _, _ = self.slot_attn(encoded, self.skip_norm)
        if self.use_conditional_slot_pruning:
            active_mask = build_active_slot_mask(
                attention,
                cov_rho=self.cov_rho,
                cov_tau=self.cov_tau,
                cov_kmin=self.cov_kmin,
                cov_novelty_alpha=self.cov_novelty_alpha,
            )
        else:
            active_mask = torch.ones(
                slots.shape[:2], device=slots.device, dtype=torch.bool
            )
        return slots, active_mask


def add_slot_extractor_args(
    parser: argparse.ArgumentParser,
    checkpoint_required: bool = True,
) -> None:
    parser.add_argument(
        "--qasa_checkpoint", type=str, required=checkpoint_required
    )
    parser.add_argument(
        "--qasa_args_json",
        type=str,
        default=None,
        help="Optional JSON containing the QASA architecture arguments.",
    )
    parser.add_argument("--which_encoder", choices=tuple(PATCH_SIZES), default=None)
    parser.add_argument("--num_slots", type=int, default=None)
    parser.add_argument("--slot_size", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--mlp_hidden_size", type=int, default=None)
    parser.add_argument("--pos_channels", type=int, default=None)
    parser.add_argument(
        "--truncate",
        choices=("bi-level", "fixed-point", "none"),
        default=None,
    )
    parser.add_argument(
        "--init_method", choices=("embedding", "shared_gaussian"), default=None
    )
    parser.add_argument("--encoder_final_norm", action="store_true", default=None)
    parser.add_argument("--skip_norm", action="store_true", default=None)
    parser.add_argument(
        "--use_conditional_slot_pruning",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--cov_rho", type=float, default=None)
    parser.add_argument("--cov_tau", type=float, default=None)
    parser.add_argument("--cov_kmin", type=int, default=None)
    parser.add_argument("--cov_novelty_alpha", type=float, default=None)


def _namespace_from_mapping(mapping: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dict(mapping))


def _load_args_from_checkpoint(checkpoint: Mapping[str, Any]) -> Optional[SimpleNamespace]:
    for key in ("args", "config", "cfg"):
        value = checkpoint.get(key)
        if isinstance(value, argparse.Namespace):
            return SimpleNamespace(**vars(value))
        if isinstance(value, Mapping):
            return _namespace_from_mapping(value)
    return None


def _load_args_from_json(path: Optional[str]) -> Optional[SimpleNamespace]:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"QASA args JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping) and isinstance(data.get("args"), Mapping):
        data = data["args"]
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return _namespace_from_mapping(data)


def _update_config(config: Dict[str, Any], namespace: Optional[SimpleNamespace]) -> None:
    if namespace is None:
        return
    for key in ARCH_KEYS:
        value = getattr(namespace, key, None)
        if value is not None:
            config[key] = value


def _normalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def _infer_slot_shape(
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Optional[int], Optional[int]]:
    embedding = state_dict.get("slot_attn.slots_init.weight")
    if embedding is not None:
        num_slots, slot_size = embedding.shape
        return int(num_slots), int(slot_size)
    slot_mu = state_dict.get("slot_attn.slot_mu")
    if slot_mu is not None:
        return None, int(slot_mu.shape[-1])
    return None, None


def _infer_encoder(state_dict: Mapping[str, torch.Tensor]) -> Optional[str]:
    layer_norm = state_dict.get("slot_attn.layer_norm.weight")
    if layer_norm is None:
        return None
    input_dim = int(layer_norm.numel())
    matches = [name for name, dimension in ENCODER_DIMS.items() if dimension == input_dim]
    return matches[0] if len(matches) == 1 else None


def _build_config(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> SimpleNamespace:
    config = dict(ARCH_DEFAULTS)
    _update_config(config, _load_args_from_json(args.qasa_args_json))
    _update_config(config, _load_args_from_checkpoint(checkpoint))
    _update_config(config, SimpleNamespace(**vars(args)))

    inferred_num_slots, inferred_slot_size = _infer_slot_shape(state_dict)
    inferred_encoder = _infer_encoder(state_dict)
    if config["num_slots"] is None:
        config["num_slots"] = inferred_num_slots
    if config["slot_size"] is None:
        config["slot_size"] = inferred_slot_size
    if config["which_encoder"] is None:
        config["which_encoder"] = inferred_encoder
    if config["init_method"] is None:
        config["init_method"] = (
            "shared_gaussian" if "slot_attn.slot_mu" in state_dict else "embedding"
        )

    if config["num_slots"] is None:
        raise ValueError("Cannot infer num_slots; pass --num_slots.")
    if config["slot_size"] is None:
        raise ValueError("Cannot infer slot_size; pass --slot_size.")
    if config["which_encoder"] is None:
        raise ValueError("Cannot infer the encoder; pass --which_encoder.")
    if inferred_num_slots is not None and int(config["num_slots"]) != inferred_num_slots:
        raise ValueError(
            f"num_slots={config['num_slots']} does not match checkpoint "
            f"num_slots={inferred_num_slots}."
        )
    if inferred_slot_size is not None and int(config["slot_size"]) != inferred_slot_size:
        raise ValueError(
            f"slot_size={config['slot_size']} does not match checkpoint "
            f"slot_size={inferred_slot_size}."
        )
    if inferred_encoder is not None and config["which_encoder"] != inferred_encoder:
        raise ValueError(
            f"which_encoder={config['which_encoder']} does not match checkpoint "
            f"encoder={inferred_encoder}."
        )
    return SimpleNamespace(**config)


def build_slot_model(args: argparse.Namespace) -> Tuple[QASASlotExtractor, int]:
    checkpoint = load_checkpoint(args.qasa_checkpoint)
    if not isinstance(checkpoint, Mapping):
        checkpoint = {"model": checkpoint}
    state_dict = extract_state_dict(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError("The QASA checkpoint does not contain a state dictionary.")
    state_dict = _normalize_state_dict(state_dict)
    config = _build_config(args, checkpoint, state_dict)

    encoder, _ = build_encoder(config.which_encoder, int(config.image_size))
    slot_attn = SlotAttentionEncoder(
        num_iterations=int(config.num_iterations),
        num_slots=int(config.num_slots),
        input_channels=ENCODER_DIMS[config.which_encoder],
        slot_size=int(config.slot_size),
        mlp_hidden_size=int(config.mlp_hidden_size),
        pos_channels=int(config.pos_channels),
        truncate=config.truncate,
        init_method=config.init_method,
    )
    model = QASASlotExtractor(encoder, slot_attn, config)

    component_state = {
        key: value
        for key, value in state_dict.items()
        if key.startswith(("encoder.", "slot_attn."))
    }
    model.load_state_dict(component_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for key in ("num_slots", "slot_size", "which_encoder"):
        setattr(args, key, getattr(config, key))
    return model, int(config.slot_size)
