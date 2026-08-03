"""Evaluation entry point for gated QASA checkpoints."""

import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from checkpoints import extract_state_dict, load_checkpoint
from datasets import COCO2017, MOVi, PascalVOC
from encoders import PATCH_SIZES, build_encoder
from qasa import QASA
from ocl_metrics import ARIMetric, UnsupervisedMaskIoUMetric
from qasa_utils import bool_flag, inv_normalize, visualize_three


def get_args_parser():
    parser = argparse.ArgumentParser("QASA evaluation", add_help=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_image_size", type=int, default=224)
    parser.add_argument("--val_mask_size", type=int, default=320)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--log_path", default="logs/qasa_eval")
    parser.add_argument("--dataset", choices=("coco", "voc", "movi"), default="coco")
    parser.add_argument("--data_path", required=True)

    parser.add_argument("--num_dec_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--num_cross_heads", type=int)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--num_slots", type=int, default=7)
    parser.add_argument("--slot_size", type=int, default=256)
    parser.add_argument("--mlp_hidden_size", type=int, default=1024)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--pos_channels", type=int, default=4)
    parser.add_argument("--dec_type", choices=("transformer", "mlp"), default="transformer")
    parser.add_argument("--mlp_dec_hidden", type=int, default=2048)
    parser.add_argument("--which_encoder", choices=tuple(PATCH_SIZES), default="dino_vitb16")
    parser.add_argument("--finetune_blocks_after", type=int, default=100)
    parser.add_argument("--encoder_final_norm", type=bool_flag, default=False)
    parser.add_argument("--truncate", choices=("bi-level", "fixed-point", "none"), default="none")
    parser.add_argument("--init_method", choices=("embedding", "shared_gaussian"), default="embedding")
    parser.add_argument(
        "--train_permutations", choices=("standard", "random", "all"), default="random"
    )
    parser.add_argument(
        "--eval_permutations", choices=("standard", "random", "all"), default="standard"
    )

    parser.add_argument("--livis", type=bool_flag, default=False)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--pre_argmax_slot_maxnorm", type=bool_flag, default=False)
    parser.add_argument("--eval_gate_slots_attn", type=bool_flag, default=False)
    parser.add_argument("--slot_minpeak", type=float, default=-1.0)
    parser.add_argument("--use_conditional_slot_pruning", action="store_true")
    parser.add_argument("--cov_rho", type=float, default=0.8)
    parser.add_argument("--cov_tau", type=float, default=0.5)
    parser.add_argument("--cov_kmin", type=int, default=2)
    parser.add_argument("--cov_novelty_alpha", type=float)
    parser.add_argument("--gate_eps", type=float, default=1e-3)
    parser.add_argument("--gate_layers", choices=("last", "all"), default="last")
    return parser


def build_dataset(args):
    if args.dataset == "voc":
        return PascalVOC(
            root=args.data_path,
            split="val",
            image_size=args.val_image_size,
            mask_size=args.val_mask_size,
        )
    if args.dataset == "coco":
        return COCO2017(
            root=args.data_path,
            livis=args.livis,
            split="val",
            image_size=args.val_image_size,
            mask_size=args.val_mask_size,
        )
    return MOVi(
        root=os.path.join(args.data_path, "validation"),
        split="validation",
        image_size=args.val_image_size,
        mask_size=args.val_mask_size,
    )


def make_metrics(device):
    return {
        "mbo_c": UnsupervisedMaskIoUMetric(
            matching="best_overlap", ignore_background=True, ignore_overlaps=True
        ).to(device),
        "mbo_i": UnsupervisedMaskIoUMetric(
            matching="best_overlap", ignore_background=True, ignore_overlaps=True
        ).to(device),
        "miou": UnsupervisedMaskIoUMetric(
            matching="hungarian", ignore_background=True, ignore_overlaps=True
        ).to(device),
        "ari": ARIMetric(foreground=True, ignore_overlaps=True).to(device),
        "mbo_c_slot": UnsupervisedMaskIoUMetric(
            matching="best_overlap", ignore_background=True, ignore_overlaps=True
        ).to(device),
        "mbo_i_slot": UnsupervisedMaskIoUMetric(
            matching="best_overlap", ignore_background=True, ignore_overlaps=True
        ).to(device),
        "miou_slot": UnsupervisedMaskIoUMetric(
            matching="hungarian", ignore_background=True, ignore_overlaps=True
        ).to(device),
        "ari_slot": ARIMetric(foreground=True, ignore_overlaps=True).to(device),
    }


def slotwise_maxnorm(attention, min_peak=-1.0, eps=1e-8):
    batch_size, num_slots, _, _ = attention.shape
    if min_peak >= 0:
        peaks = attention.view(batch_size, num_slots, -1).amax(dim=-1)
        attention = attention.masked_fill(
            (peaks < min_peak).unsqueeze(-1).unsqueeze(-1), float("-inf")
        )

    maxima = attention.view(batch_size, num_slots, -1).amax(dim=-1)
    finite = torch.isfinite(maxima)
    denominator = torch.where(finite, maxima, torch.ones_like(maxima))
    output = attention / (denominator.unsqueeze(-1).unsqueeze(-1) + eps)
    return output.masked_fill(
        (~finite).unsqueeze(-1).unsqueeze(-1), float("-inf")
    )


@torch.no_grad()
def evaluate(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    encoder, args.max_tokens = build_encoder(args.which_encoder, args.val_image_size)
    args.num_cross_heads = args.num_cross_heads or args.num_heads
    model = QASA(encoder.eval(), args)
    checkpoint = load_checkpoint(args.checkpoint_path)
    model.load_compatible_state_dict(extract_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}", flush=True)

    visualization_dir = None
    if args.save_viz:
        run_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint_path)))
        visualization_dir = os.path.join(args.log_path, run_name, "masked_image")
        os.makedirs(visualization_dir, exist_ok=True)

    gate_wp = not args.use_conditional_slot_pruning
    metrics = make_metrics(device)
    total_mse = 0.0
    mask_size = (args.val_mask_size, args.val_mask_size)

    for batch_index, (img_ids, image, true_i, true_c, mask_ignore) in enumerate(loader):
        image = image.to(device, non_blocking=True)
        true_i = true_i.to(device, non_blocking=True)
        true_c = true_c.to(device, non_blocking=True)
        mask_ignore = mask_ignore.to(device, non_blocking=True)
        loss, slot_attn, decoder_attn, _, _, _, gate = model(image, gate_wp=gate_wp)
        total_mse += loss.item()

        slot_attn = F.interpolate(slot_attn, size=mask_size, mode="bilinear")
        decoder_attn = F.interpolate(decoder_attn, size=mask_size, mode="bilinear")
        slot_scores = slot_attn.clone()
        decoder_scores = decoder_attn.clone()
        if args.eval_gate_slots_attn and gate is not None:
            active = (gate > 0.5).unsqueeze(-1).unsqueeze(-1)
            slot_scores = slot_scores.masked_fill(~active, float("-inf"))
        if args.pre_argmax_slot_maxnorm:
            slot_scores = slotwise_maxnorm(slot_scores, args.slot_minpeak)
            decoder_scores = slotwise_maxnorm(decoder_scores, args.slot_minpeak)

        pred_slot = slot_scores.argmax(dim=1)
        pred_decoder = decoder_scores.argmax(dim=1)
        true_i_oh = F.one_hot(true_i).float().permute(0, 3, 1, 2)
        true_c_oh = F.one_hot(true_c).float().permute(0, 3, 1, 2)
        pred_d_oh = F.one_hot(pred_decoder).float().permute(0, 3, 1, 2)
        pred_s_oh = F.one_hot(pred_slot).float().permute(0, 3, 1, 2)

        metrics["mbo_i"].update(pred_d_oh, true_i_oh, mask_ignore)
        metrics["mbo_c"].update(pred_d_oh, true_c_oh, mask_ignore)
        metrics["miou"].update(pred_d_oh, true_i_oh, mask_ignore)
        metrics["ari"].update(pred_d_oh, true_i_oh, mask_ignore)
        metrics["mbo_i_slot"].update(pred_s_oh, true_i_oh, mask_ignore)
        metrics["mbo_c_slot"].update(pred_s_oh, true_c_oh, mask_ignore)
        metrics["miou_slot"].update(pred_s_oh, true_i_oh, mask_ignore)
        metrics["ari_slot"].update(pred_s_oh, true_i_oh, mask_ignore)

        if visualization_dir:
            images = F.interpolate(inv_normalize(image), size=mask_size, mode="bilinear")
            visuals = visualize_three(
                images, true_i, pred_slot, N=min(32, image.shape[0])
            ).view(-1, 3, 3, *mask_size)
            for index, sample in enumerate(visuals):
                save_image(
                    make_grid(sample, nrow=3),
                    os.path.join(visualization_dir, f"{int(img_ids[index])}.png"),
                )

        if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
            print(
                f"batch={batch_index + 1}/{len(loader)} time={datetime.now():%H:%M:%S}",
                flush=True,
            )

    results = {"mse": total_mse / max(1, len(loader))}
    results.update({name: 100 * metric.compute().item() for name, metric in metrics.items()})
    print(" ".join(f"{name}={value:.4f}" for name, value in results.items()), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("QASA", parents=[get_args_parser()])
    evaluate(parser.parse_args())
