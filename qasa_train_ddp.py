"""Distributed training entry point for the baseline gated QASA model."""

import argparse
import math
import os
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from checkpoints import extract_state_dict, load_checkpoint
from datasets import COCO2017, MOVi, PascalVOC
from encoders import PATCH_SIZES, build_encoder
from qasa import QASA
from ocl_metrics import ARIMetric, UnsupervisedMaskIoUMetric
from qasa_utils import bool_flag, cosine_scheduler, inv_normalize, visualize


def get_args_parser():
    parser = argparse.ArgumentParser("QASA training", add_help=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--clip", type=float, default=0.3)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_image_size", type=int, default=224)
    parser.add_argument("--val_mask_size", type=int, default=320)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_viz_percent", type=float, default=0.2)

    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--log_path", default="logs")
    parser.add_argument("--dataset", choices=("coco", "voc", "movi"), default="coco")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", default="train")

    parser.add_argument("--lr_main", type=float, default=4e-4)
    parser.add_argument("--lr_min", type=float, default=4e-7)
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

    parser.add_argument("--use_conditional_slot_pruning", action="store_true")
    parser.add_argument("--cov_rho", type=float, default=0.8)
    parser.add_argument("--cov_tau", type=float, default=0.5)
    parser.add_argument("--cov_kmin", type=int, default=2)
    parser.add_argument("--cov_novelty_alpha", type=float)
    parser.add_argument("--gate_eps", type=float, default=1e-3)
    parser.add_argument("--gate_layers", choices=("last", "all"), default="last")
    parser.add_argument("--gate_warmup", type=int, default=0)
    return parser


def build_datasets(args):
    if args.dataset == "voc":
        train_dataset = PascalVOC(
            root=args.data_path,
            split="trainaug",
            image_size=args.image_size,
            mask_size=args.image_size,
        )
        val_dataset = PascalVOC(
            root=args.data_path,
            split="val",
            image_size=args.val_image_size,
            mask_size=args.val_mask_size,
        )
    elif args.dataset == "coco":
        train_dataset = COCO2017(
            root=args.data_path,
            split=args.split,
            image_size=args.image_size,
            mask_size=args.image_size,
        )
        val_dataset = COCO2017(
            root=args.data_path,
            split="val",
            image_size=args.val_image_size,
            mask_size=args.val_mask_size,
        )
    else:
        train_dataset = MOVi(
            root=os.path.join(args.data_path, "train"),
            split="train",
            image_size=args.image_size,
            mask_size=args.image_size,
            frames_per_clip=9,
        )
        val_dataset = MOVi(
            root=os.path.join(args.data_path, "validation"),
            split="validation",
            image_size=args.val_image_size,
            mask_size=args.val_mask_size,
        )
    return train_dataset, val_dataset


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


def restore_training_state(model, checkpoint_path):
    defaults = {
        "epoch": 0,
        "best_val_loss": math.inf,
        "best_val_ari": 0,
        "best_val_ari_slot": 0,
        "best_mbo_c": 0,
        "best_mbo_i": 0,
        "best_miou": 0,
        "best_mbo_c_slot": 0,
        "best_mbo_i_slot": 0,
        "best_miou_slot": 0,
        "best_epoch": 0,
    }
    if not checkpoint_path:
        return None, defaults
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)
    model.load_compatible_state_dict(extract_state_dict(checkpoint), strict=True)
    state = {key: checkpoint.get(key, value) for key, value in defaults.items()}
    return checkpoint, state


def train(args):
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    torch.manual_seed(args.seed)

    log_dir = os.path.join(args.log_path, datetime.today().strftime("%y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir) if rank == 0 else None
    if writer is not None:
        writer.add_text("hparams", "__".join(f"{k}={v}" for k, v in vars(args).items()))

    train_dataset, val_dataset = build_datasets(args)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=True,
        batch_size=args.batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        drop_last=False,
        batch_size=args.eval_batch_size,
        **loader_kwargs,
    )

    encoder, args.max_tokens = build_encoder(args.which_encoder, args.val_image_size)
    encoder = encoder.to(device).eval()
    args.num_cross_heads = args.num_cross_heads or args.num_heads
    model = QASA(encoder, args)
    checkpoint, state = restore_training_state(model, args.checkpoint_path)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr_main,
    )
    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    lr_schedule = cosine_scheduler(
        args.lr_main,
        args.lr_min,
        args.epochs,
        len(train_loader),
        warmup_epochs=5,
        start_warmup_value=0,
    )
    metrics = make_metrics(device)
    visualization_interval = max(1, int(args.epochs * args.eval_viz_percent))

    for epoch in range(state["epoch"], args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        gate_wp = epoch < args.gate_warmup

        for batch_index, image in enumerate(train_loader):
            image = image.to(device, non_blocking=True)
            global_step = epoch * len(train_loader) + batch_index
            optimizer.param_groups[0]["lr"] = lr_schedule[global_step]
            optimizer.zero_grad()
            loss, *_ = model(image, gate_wp=gate_wp)
            loss = loss.mean()
            loss.backward()
            total_norm = clip_grad_norm_(model.parameters(), args.clip, "inf")
            optimizer.step()

            if rank == 0 and batch_index % 100 == 0:
                print(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                    f"epoch={epoch + 1} batch={batch_index}/{len(train_loader)} "
                    f"lr={optimizer.param_groups[0]['lr']:.6g} "
                    f"mse={loss.item():.6f} norm={total_norm.item():.6f}",
                    flush=True,
                )
                writer.add_scalar("TRAIN/mse", loss.item(), global_step)
                writer.add_scalar("TRAIN/lr_main", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("TRAIN/total_norm", total_norm.item(), global_step)

        values, last_batch = validate(model, val_loader, metrics, device, args)
        val_loss = values["mse"]
        if rank == 0:
            print_validation(epoch, values)
            for name, value in values.items():
                writer.add_scalar(f"VAL/{name}", value, epoch + 1)

            improved = (
                state["best_mbo_c_slot"] < values["mbo_c_slot"]
                or state["best_mbo_i_slot"] < values["mbo_i_slot"]
            )
            if improved:
                state.update(
                    best_val_loss=val_loss,
                    best_val_ari=values["ari"],
                    best_val_ari_slot=values["ari_slot"],
                    best_mbo_c=values["mbo_c"],
                    best_mbo_i=values["mbo_i"],
                    best_miou=values["miou"],
                    best_mbo_c_slot=values["mbo_c_slot"],
                    best_mbo_i_slot=values["mbo_i_slot"],
                    best_miou_slot=values["miou_slot"],
                    best_epoch=epoch + 1,
                )
                torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))

            if epoch % visualization_interval == 0 or epoch == args.epochs - 1:
                add_visualization(writer, epoch, last_batch, args)

            saved_state = dict(state)
            saved_state["epoch"] = epoch + 1
            saved_state["model"] = model.state_dict()
            saved_state["optimizer"] = optimizer.state_dict()
            torch.save(saved_state, os.path.join(log_dir, "checkpoint.pt.tar"))
            print(
                f"best validation loss={float(state['best_val_loss']):.6f} "
                f"at epoch {state['best_epoch']}",
                flush=True,
            )

    if writer is not None:
        writer.close()
    dist.destroy_process_group()


@torch.no_grad()
def validate(model, val_loader, metrics, device, args):
    model.eval()
    total_mse = 0.0
    last_batch = None
    for img_ids, image, true_mask_i, true_mask_c, mask_ignore in val_loader:
        image = image.to(device, non_blocking=True)
        true_mask_i = true_mask_i.to(device, non_blocking=True)
        true_mask_c = true_mask_c.to(device, non_blocking=True)
        mask_ignore = mask_ignore.to(device, non_blocking=True)
        loss, slot_attn, decoder_attn, *_ = model(image)
        total_mse += loss.mean().item()

        slot_attn = F.interpolate(slot_attn, size=args.val_mask_size, mode="bilinear")
        decoder_attn = F.interpolate(decoder_attn, size=args.val_mask_size, mode="bilinear")
        pred_slot = slot_attn.argmax(1)
        pred_decoder = decoder_attn.argmax(1)
        true_i = F.one_hot(true_mask_i).float().permute(0, 3, 1, 2)
        true_c = F.one_hot(true_mask_c).float().permute(0, 3, 1, 2)
        pred_d = F.one_hot(pred_decoder).float().permute(0, 3, 1, 2)
        pred_s = F.one_hot(pred_slot).float().permute(0, 3, 1, 2)

        metrics["mbo_i"].update(pred_d, true_i, mask_ignore)
        metrics["mbo_c"].update(pred_d, true_c, mask_ignore)
        metrics["miou"].update(pred_d, true_i, mask_ignore)
        metrics["ari"].update(pred_d, true_i, mask_ignore)
        metrics["mbo_i_slot"].update(pred_s, true_i, mask_ignore)
        metrics["mbo_c_slot"].update(pred_s, true_c, mask_ignore)
        metrics["miou_slot"].update(pred_s, true_i, mask_ignore)
        metrics["ari_slot"].update(pred_s, true_i, mask_ignore)
        last_batch = (image, true_mask_i, true_mask_c, pred_decoder, pred_slot, decoder_attn, slot_attn)

    values = {"mse": total_mse / max(1, len(val_loader))}
    for name, metric in metrics.items():
        values[name] = 100 * metric.compute().item()
        metric.reset()
    return values, last_batch


def print_validation(epoch, values):
    rendered = " ".join(f"{key}={float(value):.4f}" for key, value in values.items())
    print(f"validation epoch={epoch + 1} {rendered}", flush=True)


def add_visualization(writer, epoch, batch, args):
    image, true_i, true_c, pred_decoder, pred_slot, decoder_attn, slot_attn = batch
    image = F.interpolate(inv_normalize(image), size=args.val_mask_size, mode="bilinear")
    decoder_rgb = image.unsqueeze(1) * decoder_attn.unsqueeze(2) + 1.0 - decoder_attn.unsqueeze(2)
    slot_rgb = image.unsqueeze(1) * slot_attn.unsqueeze(2) + 1.0 - slot_attn.unsqueeze(2)
    is_instance = args.dataset == "movi"
    ground_truth = true_i if is_instance else true_c
    visual = visualize(
        image,
        ground_truth,
        pred_decoder,
        decoder_rgb,
        pred_slot,
        slot_rgb,
        N=32,
        is_ins=is_instance,
    )
    grid = vutils.make_grid(visual, nrow=2 * args.num_slots + 4, pad_value=0.2)[:, 2:-2, 2:-2]
    grid = F.interpolate(grid.unsqueeze(1), scale_factor=0.3, mode="bilinear").squeeze()
    writer.add_image(f"VAL_recon/epoch={epoch + 1:03}", grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("QASA", parents=[get_args_parser()])
    train(parser.parse_args())
