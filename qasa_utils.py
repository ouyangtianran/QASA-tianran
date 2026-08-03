'''Utilities based on the SLATE and OCLF implementations:
https://github.com/singhgautam/slate/blob/master/utils.py
https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/utils/masking.py
'''
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080','#C56932',
'#b7a58c', '#3a627d', '#9abc15', '#54810c', '#a7389c', '#687253', '#61f584', '#9a17d4', '#52b0c1', '#21f5b4', '#a2856c', '#9b1c34', '#4b1062', '#7cf406', '#0b1f63']

def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):

    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gru_cell(input_size, hidden_size, bias=True):

    m = nn.GRUCell(input_size, hidden_size, bias)

    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)

    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

    return m

inv_normalize = transforms.Compose([transforms.Normalize((0., 0., 0.), (1/0.229, 1/0.224, 1/0.225)),
                                    transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1))])


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    total_iters = epochs * niter_per_ep
    if total_iters <= 0:
        return np.array([])

    warmup_schedule = np.array([])
    warmup_iters = min(warmup_epochs * niter_per_ep, total_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    remaining_iters = total_iters - warmup_iters
    if remaining_iters > 0:
        iters = np.arange(remaining_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    else:
        schedule = np.array([])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def spiral_pattern(A, how = 'left_top'):

    out = []

    if how == 'left_top':
        A = np.array(A)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise

    if how == 'top_left':
        A = np.rot90(np.fliplr(np.array(A)), k=1)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise

    if how == 'right_top':
        A = np.fliplr(np.array(A))
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise

    if how == 'top_right':
        A = np.rot90(np.array(A), k=1)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise

    if how == 'right_bottom':
        A = np.rot90(np.array(A), k=2)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise

    if how == 'bottom_right':
        A = np.fliplr(np.rot90(np.array(A), k=1))
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise

    if how == 'left_bottom':
        A = np.rot90(np.fliplr(np.array(A)), k=2)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise

    if how == 'bottom_left':
        A = np.rot90(np.array(A), k=3)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)

def visualize(image, true_mask, pred_dec_mask, rgb_dec_attns, pred_default_mask, rgb_default_attns, N=8, is_ins=False):
    import colorsys
    B = min(int(N), int(image.size(0)))
    image = image[:B]
    pred_dec_mask = pred_dec_mask[:B]
    pred_default_mask = pred_default_mask[:B]
    rgb_dec_attns = rgb_dec_attns[:B]
    rgb_default_attns = rgb_default_attns[:B]
    true_mask = true_mask[:B]

    _, _, H, W = image.shape

    def _ensure_colors(num_masks, base_colors):
        palette = list(base_colors)
        if len(palette) >= num_masks:
            return palette
        need = num_masks - len(palette)
        for i in range(need):
            h = (i + 1) / (need + 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
            palette.append((int(r*255), int(g*255), int(b*255)))
        return palette

    def _filter_empty(masks_bool: torch.Tensor) -> torch.Tensor:
        # masks_bool: [C,H,W] (bool)
        if masks_bool.numel() == 0:
            return masks_bool
        keep = masks_bool.flatten(1).any(dim=1)
        return masks_bool[keep]

    def _draw_one(img_u8: torch.Tensor, masks_bool: torch.Tensor):
        masks_bool = _filter_empty(masks_bool)
        if masks_bool.numel() == 0 or masks_bool.shape[0] == 0:
            return img_u8
        cols = _ensure_colors(masks_bool.shape[0], colors)
        return draw_segmentation_masks(img_u8, masks=masks_bool, alpha=.5, colors=cols)

    rgb_pred_dec_list = []
    for idx in range(B):
        masks_dec = torch.nn.functional.one_hot(pred_dec_mask[idx]).permute(2,0,1).to(torch.bool).cpu()
        img_u8 = (image[idx]*255).to(torch.uint8).cpu()
        rgb_pred_dec_list.append(_draw_one(img_u8, masks_dec))
    rgb_pred_dec_mask = (torch.stack(rgb_pred_dec_list) / 255.)

    rgb_pred_def_list = []
    for idx in range(B):
        masks_def = torch.nn.functional.one_hot(pred_default_mask[idx]).permute(2,0,1).to(torch.bool).cpu()
        img_u8 = (image[idx]*255).to(torch.uint8).cpu()
        rgb_pred_def_list.append(_draw_one(img_u8, masks_def))
    rgb_pred_default_mask = (torch.stack(rgb_pred_def_list) / 255.)

    # ------- GT -------
    if is_ins:
        def _label_to_bool_channels(lbl: torch.Tensor):
            u, inv = torch.unique(lbl, return_inverse=True)
            k = int(u.numel())
            if k <= 1:
                return torch.zeros((0, lbl.shape[0], lbl.shape[1]), dtype=torch.bool)
            return torch.nn.functional.one_hot(inv.view(lbl.shape[0], lbl.shape[1]), num_classes=k).permute(2,0,1).to(torch.bool)

        rgb_true_list = []
        for idx in range(B):
            masks_gt = _label_to_bool_channels(true_mask[idx]).cpu()
            img_u8 = (image[idx]*255).to(torch.uint8).cpu()
            rgb_true_list.append(_draw_one(img_u8, masks_gt))
        rgb_true_mask = (torch.stack(rgb_true_list) / 255.)
    else:
        rgb_true_list = []
        for idx in range(B):
            lbl = true_mask[idx]
            u, inv = torch.unique(lbl, return_inverse=True)
            k = int(u.numel())
            if k <= 1:
                masks_gt = torch.zeros((0, lbl.shape[0], lbl.shape[1]), dtype=torch.bool)
            else:
                masks_gt = torch.nn.functional.one_hot(inv.view(lbl.shape[0], lbl.shape[1]), num_classes=k).permute(2,0,1).to(torch.bool)
            img_u8 = (image[idx]*255).to(torch.uint8).cpu()
            rgb_true_list.append(_draw_one(img_u8, masks_gt))
        rgb_true_mask = (torch.stack(rgb_true_list) / 255.)

    image_vis = image.expand(-1, 3, H, W).unsqueeze(1).cpu()           # [B,1,3,H,W]
    rgb_default_attns = rgb_default_attns.expand(-1, -1, 3, H, W).cpu() # [B,S,3,H,W]
    rgb_dec_attns = rgb_dec_attns.expand(-1, -1, 3, H, W).cpu()         # [B,S,3,H,W]

    rgb_true_mask = rgb_true_mask.unsqueeze(dim=1).cpu()                # [B,1,3,H,W]
    rgb_pred_default_mask = rgb_pred_default_mask.unsqueeze(dim=1).cpu()# [B,1,3,H,W]
    rgb_pred_dec_mask = rgb_pred_dec_mask.unsqueeze(dim=1).cpu()        # [B,1,3,H,W]

    return torch.cat(
        (image_vis, rgb_true_mask, rgb_pred_dec_mask, rgb_dec_attns, rgb_pred_default_mask, rgb_default_attns),
        dim=1
    ).view(-1, 3, H, W)

def visualize_three(
    image,
    true_mask,        # [B,H,W]    long
    pred_mask,
    N=8,
    ignore_labels=(255,),
    alpha=0.5
):
    import colorsys
    B = min(int(N), int(image.size(0)))
    image = image[:B]
    true_mask = true_mask[:B]
    pred_mask = pred_mask[:B]
    _, _, H, W = image.shape

    base_colors = list(globals().get('colors', []))
    def _ensure_colors(num_masks: int):
        palette = list(base_colors)
        if len(palette) >= num_masks:
            return palette
        need = num_masks - len(palette)
        for i in range(need):
            h = (i + 1) / (need + 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
            palette.append((int(r*255), int(g*255), int(b*255)))
        return palette

    def _label_to_masks_bool(lbl: torch.Tensor, ignore_set: set):
        uniq = torch.unique(lbl)
        masks = []
        for v in uniq.tolist():
            if v in ignore_set:
                continue
            m = (lbl == v)
            if m.any():
                masks.append(m)
        if len(masks) == 0:
            return torch.zeros((0, H, W), dtype=torch.bool)
        return torch.stack(masks, dim=0)  # [C',H,W]

    def _draw_overlay(img_f01: torch.Tensor, lbl_hw: torch.Tensor):
        masks = _label_to_masks_bool(lbl_hw, set(ignore_labels))
        img_u8 = (img_f01 * 255).to(torch.uint8).cpu()
        if masks.numel() == 0 or masks.shape[0] == 0:
            return img_u8
        cols = _ensure_colors(masks.shape[0])
        return draw_segmentation_masks(img_u8, masks=masks.cpu(), alpha=alpha, colors=cols)

    image_vis = image.expand(-1, 3, H, W).unsqueeze(1).cpu()          # [B,1,3,H,W]

    gt_imgs = [ _draw_overlay(image[i].cpu(), true_mask[i].cpu()) for i in range(B) ]
    rgb_true = (torch.stack(gt_imgs) / 255.).unsqueeze(1).cpu()       # [B,1,3,H,W]

    pred_imgs = [ _draw_overlay(image[i].cpu(), pred_mask[i].cpu()) for i in range(B) ]
    rgb_pred = (torch.stack(pred_imgs) / 255.).unsqueeze(1).cpu()     # [B,1,3,H,W]

    out = torch.cat((image_vis, rgb_true, rgb_pred), dim=1)           # [B,3,3,H,W]
    return out.view(-1, 3, H, W)
