"""Adversarial losses."""

from __future__ import annotations

import torch
from torch import nn


class HingeDiscriminatorLoss(nn.Module):
    def forward(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


class MultiScaleHingeDiscriminatorLoss(HingeDiscriminatorLoss):
    pass


class MultiScaleHingeGeneratorLoss(nn.Module):
    def forward(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()


class DiscriminatorFeatureMatchingLoss(nn.Module):
    def forward(self, real_features: list[torch.Tensor], fake_features: list[torch.Tensor]) -> torch.Tensor:
        if not real_features or not fake_features:
            return torch.tensor(0.0)
        return sum(torch.mean(torch.abs(a.detach() - b)) for a, b in zip(real_features, fake_features))
