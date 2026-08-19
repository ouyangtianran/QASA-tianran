"""Train and evaluate a COCO property probe on frozen QASA slots."""

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from checkpoints import load_checkpoint
from set_prediction.slot_extractor import add_slot_extractor_args, build_slot_model


COCO_CATEGORY_IDS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def deterministic_transform(
    image: Image.Image,
    instance_mask: Image.Image,
    class_mask: Image.Image,
    ignore_mask: Image.Image,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image = TF.resize(image, image_size, interpolation=InterpolationMode.BILINEAR)
    image = TF.center_crop(image, [image_size, image_size])
    image = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)

    def transform_mask(mask: Image.Image) -> torch.Tensor:
        mask = TF.resize(mask, image_size, interpolation=InterpolationMode.NEAREST)
        mask = TF.center_crop(mask, [image_size, image_size])
        return TF.pil_to_tensor(mask).squeeze(0).long()

    return (
        image,
        transform_mask(instance_mask),
        transform_mask(class_mask),
        transform_mask(ignore_mask),
    )


def masks_to_properties(
    instance_mask: torch.Tensor,
    class_mask: torch.Tensor,
    ignore_mask: Optional[torch.Tensor] = None,
    ignore_overlaps: bool = True,
    min_area: int = 1,
):
    height, width = instance_mask.shape
    if ignore_mask is not None:
        if ignore_mask.ndim == 3:
            ignore_mask = ignore_mask.squeeze(0)
        ignore_mask = ignore_mask != 0

    if ignore_overlaps and ignore_mask is not None:
        valid = ~ignore_mask
        instance_mask = instance_mask * valid
        class_mask = class_mask * valid

    objects = []
    instance_ids = torch.unique(instance_mask)
    for instance_id in instance_ids[instance_ids != 0].tolist():
        mask = instance_mask == instance_id
        area = int(mask.sum().item())
        if area < min_area:
            continue

        class_values = class_mask[mask]
        if class_values.numel() == 0:
            continue
        class_index = int(torch.mode(class_values).values.item())
        if class_index == 0:
            continue

        ys, xs = torch.nonzero(mask, as_tuple=True)
        if xs.numel() == 0:
            continue
        x = xs.float().mean() / (width - 1)
        y = ys.float().mean() / (height - 1)
        objects.append((area, class_index - 1, float(x), float(y)))

    if not objects:
        return None

    objects.sort(key=lambda item: item[0], reverse=True)
    return (
        torch.tensor([item[1] for item in objects], dtype=torch.long),
        torch.tensor([[item[2], item[3]] for item in objects], dtype=torch.float32),
        torch.tensor([item[0] for item in objects], dtype=torch.long),
    )


def resolve_coco_image_dir(root: str, split: str, year: str) -> str:
    candidates = (
        os.path.join(root, f"{split}{year}"),
        os.path.join(root, "images", f"{split}{year}"),
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find an image directory under {root}")


class COCOPropertyDataset(Dataset):
    def __init__(
        self,
        coco_root: str,
        split: str,
        year: str,
        image_size: int,
        image_ids: Optional[List[int]] = None,
        ignore_overlaps: bool = True,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"Unsupported COCO split: {split}")
        annotation_file = os.path.join(
            coco_root, "annotations", f"instances_{split}{year}.json"
        )
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(annotation_file)

        self.coco = COCO(annotation_file)
        self.image_dir = resolve_coco_image_dir(coco_root, split, year)
        self.ids = list(self.coco.imgs) if image_ids is None else list(image_ids)
        self.image_size = image_size
        self.ignore_overlaps = ignore_overlaps

    def __len__(self) -> int:
        return len(self.ids)

    def _build_masks(self, image_id: int):
        metadata = self.coco.loadImgs(image_id)[0]
        height, width = metadata["height"], metadata["width"]
        class_mask = np.zeros((height, width), dtype=np.uint8)
        instance_mask = np.zeros((height, width), dtype=np.uint8)
        overlap_count = np.zeros((height, width), dtype=np.uint8)

        annotation_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        instance_index = 0
        for annotation in self.coco.loadAnns(annotation_ids):
            category_id = annotation["category_id"]
            if category_id not in COCO_CATEGORY_IDS:
                continue
            class_index = COCO_CATEGORY_IDS.index(category_id)
            if class_index == 0:
                continue

            mask = self.coco.annToMask(annotation).astype(np.uint8)
            if mask.sum() == 0:
                continue
            overlap_count += mask
            instance_index += 1
            empty = instance_mask == 0
            instance_mask[empty] += (mask[empty] * instance_index).astype(np.uint8)
            class_mask[empty] += (mask[empty] * class_index).astype(np.uint8)

        ignore_mask = (overlap_count > 1).astype(np.uint8)
        return (
            Image.fromarray(instance_mask),
            Image.fromarray(class_mask),
            Image.fromarray(ignore_mask),
        )

    def __getitem__(self, index: int):
        attempts = min(20, len(self.ids))
        for offset in range(attempts):
            image_id = self.ids[(index + offset) % len(self.ids)]
            metadata = self.coco.loadImgs(image_id)[0]
            path = os.path.join(self.image_dir, metadata["file_name"])
            image = Image.open(path).convert("RGB")
            masks = self._build_masks(image_id)
            image, instance_mask, class_mask, ignore_mask = deterministic_transform(
                image, *masks, self.image_size
            )
            properties = masks_to_properties(
                instance_mask,
                class_mask,
                ignore_mask,
                ignore_overlaps=self.ignore_overlaps,
            )
            if properties is None:
                continue
            classes, coordinates, areas = properties
            return image, {
                "classes": classes,
                "coords": coordinates,
                "areas": areas,
            }
        raise RuntimeError(f"No valid object found near dataset index {index}")


def collate_variable_length(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


class LinearProbe(nn.Module):
    def __init__(self, slot_size: int, num_classes: int = 80):
        super().__init__()
        self.cls = nn.Linear(slot_size, num_classes)
        self.xy = nn.Linear(slot_size, 2)

    def forward(self, slots: torch.Tensor):
        return self.cls(slots), self.xy(slots)


class MLPProbe(nn.Module):
    def __init__(self, slot_size: int, num_classes: int = 80, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(slot_size, hidden_size)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.cls = nn.Linear(hidden_size, num_classes)
        self.xy = nn.Linear(hidden_size, 2)

    def forward(self, slots: torch.Tensor):
        hidden = self.act(self.fc1(slots))
        return self.cls(hidden), self.xy(hidden)


def hungarian_match_and_loss(
    class_logits: torch.Tensor,
    predicted_xy: torch.Tensor,
    target_classes: torch.Tensor,
    target_xy_standardized: torch.Tensor,
):
    with torch.no_grad():
        class_cost = -F.log_softmax(class_logits, dim=-1)[:, target_classes]
        position_cost = (
            (predicted_xy[:, None] - target_xy_standardized[None]) ** 2
        ).sum(dim=-1)
        row_indices, column_indices = linear_sum_assignment(
            (class_cost + position_cost).cpu().numpy()
        )

    rows = torch.as_tensor(row_indices, device=class_logits.device, dtype=torch.long)
    columns = torch.as_tensor(
        column_indices, device=class_logits.device, dtype=torch.long
    )
    loss = F.cross_entropy(class_logits[rows], target_classes[columns])
    loss += F.mse_loss(predicted_xy[rows], target_xy_standardized[columns])
    return loss, (rows, columns)


def prepare_sample(
    class_logits: torch.Tensor,
    predicted_xy: torch.Tensor,
    target: Dict[str, torch.Tensor],
    mean_xy: torch.Tensor,
    std_xy: torch.Tensor,
    device: torch.device,
):
    target_classes = target["classes"].to(device)
    target_xy = target["coords"].to(device)
    return (
        class_logits,
        predicted_xy,
        target_classes,
        target_xy,
        (target_xy - mean_xy) / std_xy,
    )


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size == 0:
        return float("nan")
    residual = ((target - prediction) ** 2).sum()
    total = ((target - target.mean()) ** 2).sum()
    return 0.0 if total < 1e-12 else float(1.0 - residual / total)


@torch.no_grad()
def evaluate_metrics(slot_model, probe, loader, mean_xy, std_xy, device):
    slot_model.eval()
    probe.eval()
    totals = np.zeros(80, dtype=np.int64)
    top1_correct = np.zeros(80, dtype=np.int64)
    top5_correct = np.zeros(80, dtype=np.int64)
    true_x, predicted_x, true_y, predicted_y = [], [], [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        slots = slot_model(images)
        batch_class_logits, batch_xy = probe(slots)

        for index, target in enumerate(targets):
            sample = prepare_sample(
                batch_class_logits[index],
                batch_xy[index],
                target,
                mean_xy,
                std_xy,
                device,
            )
            class_logits, xy, target_classes, target_xy, target_xy_std = sample
            _, (rows, columns) = hungarian_match_and_loss(
                class_logits, xy, target_classes, target_xy_std
            )
            matched_logits = class_logits[rows]
            matched_classes = target_classes[columns]
            top1 = matched_logits.argmax(dim=-1)
            top5 = torch.topk(matched_logits, k=5, dim=-1).indices

            for class_index in matched_classes.tolist():
                totals[class_index] += 1
            for item, class_index in enumerate(matched_classes.tolist()):
                top1_correct[class_index] += int(top1[item].item() == class_index)
                top5_correct[class_index] += int(class_index in top5[item].tolist())

            matched_prediction = xy[rows] * std_xy + mean_xy
            matched_target = target_xy[columns]
            true_x.append(matched_target[:, 0].cpu().numpy())
            predicted_x.append(matched_prediction[:, 0].cpu().numpy())
            true_y.append(matched_target[:, 1].cpu().numpy())
            predicted_y.append(matched_prediction[:, 1].cpu().numpy())

    present = totals > 0
    concatenate = lambda values: np.concatenate(values) if values else np.array([])
    return {
        "top1_macro": float((top1_correct[present] / totals[present]).mean())
        if present.any()
        else float("nan"),
        "top5_macro": float((top5_correct[present] / totals[present]).mean())
        if present.any()
        else float("nan"),
        "r2x": r2_score(concatenate(true_x), concatenate(predicted_x)),
        "r2y": r2_score(concatenate(true_y), concatenate(predicted_y)),
    }


@torch.no_grad()
def compute_xy_mean_std(loader, device):
    coordinates = [target["coords"] for _, targets in loader for target in targets]
    xy = torch.cat(coordinates).to(device)
    return xy.mean(dim=0), xy.std(dim=0, unbiased=False).clamp_min(1e-6)


def learning_rate(step: int, base_lr: float = 1e-3, decay_every: int = 2000):
    return base_lr * (0.5 ** (step // decay_every))


@torch.no_grad()
def evaluate_loss(slot_model, probe, loader, mean_xy, std_xy, device):
    slot_model.eval()
    probe.eval()
    total_loss = 0.0
    sample_count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        slots = slot_model(images)
        batch_class_logits, batch_xy = probe(slots)
        for index, target in enumerate(targets):
            sample = prepare_sample(
                batch_class_logits[index],
                batch_xy[index],
                target,
                mean_xy,
                std_xy,
                device,
            )
            class_logits, xy, target_classes, _, target_xy_std = sample
            loss, _ = hungarian_match_and_loss(
                class_logits, xy, target_classes, target_xy_std
            )
            total_loss += float(loss.item())
            sample_count += 1
    return total_loss / max(sample_count, 1)


def train_probe(
    slot_model,
    probe,
    train_loader,
    validation_loader,
    mean_xy,
    std_xy,
    device,
    max_steps: int,
    log_every: int,
    validation_every: int,
    output_path: str,
):
    slot_model.eval()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    best_validation_loss = float("inf")
    train_iterator = iter(train_loader)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    for step in range(max_steps):
        try:
            images, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, targets = next(train_iterator)

        probe.train()
        images = images.to(device, non_blocking=True)
        optimizer.param_groups[0]["lr"] = learning_rate(step)
        with torch.no_grad():
            slots = slot_model(images)
        batch_class_logits, batch_xy = probe(slots)

        losses = []
        for index, target in enumerate(targets):
            sample = prepare_sample(
                batch_class_logits[index],
                batch_xy[index],
                target,
                mean_xy,
                std_xy,
                device,
            )
            class_logits, xy, target_classes, _, target_xy_std = sample
            loss, _ = hungarian_match_and_loss(
                class_logits, xy, target_classes, target_xy_std
            )
            losses.append(loss)
        if not losses:
            raise RuntimeError("No object target was available for this batch.")

        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            print(
                f"step={step + 1:05d} train_loss={loss.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )

        should_validate = (step + 1) % validation_every == 0 or step + 1 == max_steps
        if should_validate:
            validation_loss = evaluate_loss(
                slot_model,
                probe,
                validation_loader,
                mean_xy,
                std_xy,
                device,
            )
            print(
                f"step={step + 1:05d} validation_loss={validation_loss:.4f} "
                f"best={best_validation_loss:.4f}",
                flush=True,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    {
                        "probe": probe.state_dict(),
                        "mean_xy": mean_xy.detach().cpu(),
                        "std_xy": std_xy.detach().cpu(),
                    },
                    output_path,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and test a set-prediction probe on frozen QASA slots."
    )
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--year", default="2017")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    add_slot_extractor_args(parser)
    parser.add_argument("--probe_type", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--validation_every", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="logs/set_prediction/probe_best.pth")
    return parser


def main():
    args = build_parser().parse_args()
    positive_values = (
        args.train_size,
        args.val_size,
        args.batch_size,
        args.max_steps,
        args.log_every,
        args.validation_every,
    )
    if min(positive_values) <= 0:
        raise ValueError("Dataset, batch, step, and interval values must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    requested_device = torch.device(args.device)
    device = requested_device
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    train_annotations = os.path.join(
        args.coco_root, "annotations", f"instances_train{args.year}.json"
    )
    validation_annotations = os.path.join(
        args.coco_root, "annotations", f"instances_val{args.year}.json"
    )
    coco_train = COCO(train_annotations)
    train_ids = list(coco_train.imgs)
    random.shuffle(train_ids)
    required_images = args.train_size + args.val_size
    if required_images > len(train_ids):
        raise ValueError(
            f"Requested {required_images} train/validation images, "
            f"but COCO contains {len(train_ids)}."
        )
    probe_train_ids = train_ids[: args.train_size]
    probe_validation_ids = train_ids[args.train_size : required_images]
    coco_validation = COCO(validation_annotations)

    print("Building frozen QASA slot extractor...", flush=True)
    slot_model, slot_size = build_slot_model(args)
    slot_model = slot_model.to(device)
    print(
        f"encoder={args.which_encoder} num_slots={args.num_slots} slot_size={slot_size}",
        flush=True,
    )

    dataset_options = {
        "coco_root": args.coco_root,
        "year": args.year,
        "image_size": args.image_size,
    }
    train_dataset = COCOPropertyDataset(
        split="train", image_ids=probe_train_ids, **dataset_options
    )
    validation_dataset = COCOPropertyDataset(
        split="train", image_ids=probe_validation_ids, **dataset_options
    )
    test_dataset = COCOPropertyDataset(
        split="val", image_ids=list(coco_validation.imgs), **dataset_options
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_variable_length,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    print("Computing training coordinate statistics...", flush=True)
    mean_xy, std_xy = compute_xy_mean_std(train_loader, device)
    probe_class = LinearProbe if args.probe_type == "linear" else MLPProbe
    probe = probe_class(slot_size).to(device)

    print("Training set-prediction probe...", flush=True)
    train_probe(
        slot_model,
        probe,
        train_loader,
        validation_loader,
        mean_xy,
        std_xy,
        device,
        max_steps=args.max_steps,
        log_every=args.log_every,
        validation_every=args.validation_every,
        output_path=args.out,
    )

    print("Evaluating the best probe on COCO val2017...", flush=True)
    probe_checkpoint = load_checkpoint(args.out)
    probe.load_state_dict(probe_checkpoint["probe"])
    mean_xy = probe_checkpoint["mean_xy"].to(device)
    std_xy = probe_checkpoint["std_xy"].to(device)
    metrics = evaluate_metrics(
        slot_model, probe, test_loader, mean_xy, std_xy, device
    )
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
