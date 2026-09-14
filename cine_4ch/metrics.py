"""Segmentation metrics for multiclass training."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from cine_4ch.config import LABEL_NAMES, NUM_CLASSES


def multiclass_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = -1,
) -> Dict[str, float]:
    """
    Compute mean Dice and per-class Dice from logits (B, C, H, W) and targets (B, H, W).
    """
    preds = logits.argmax(dim=1)
    dice_scores: Dict[str, float] = {}
    class_dices: List[float] = []

    for class_id in range(num_classes):
        pred_mask = preds == class_id
        target_mask = targets == class_id
        if ignore_index >= 0:
            valid = targets != ignore_index
            pred_mask = pred_mask & valid
            target_mask = target_mask & valid

        pred_flat = pred_mask.reshape(-1).float()
        target_flat = target_mask.reshape(-1).float()
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = ((2.0 * intersection + 1e-6) / (union + 1e-6)).item()
        dice_scores[f"dice_class_{class_id}"] = dice
        dice_scores[f"dice_{LABEL_NAMES.get(class_id, class_id)}"] = dice
        class_dices.append(dice)

    dice_scores["dice_mean"] = sum(class_dices) / len(class_dices)
    if num_classes > 1:
        fg_dices = class_dices[1:]
        dice_scores["dice_mean_foreground"] = sum(fg_dices) / len(fg_dices)
    return dice_scores


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Cross-entropy plus soft multiclass Dice loss."""
    ce = F.cross_entropy(logits, targets)

    # One-hot targets for Dice.
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    probs = F.softmax(logits, dim=1)
    dims = (0, 2, 3)
    intersection = (probs * targets_one_hot).sum(dims)
    union = probs.sum(dims) + targets_one_hot.sum(dims)
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    dice_loss = 1.0 - dice.mean()

    return ce_weight * ce + dice_weight * dice_loss
