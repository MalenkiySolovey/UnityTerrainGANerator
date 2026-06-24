"""SPADE terrain GAN training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from terrain_gan.data.dataset import TerrainBatch
from terrain_gan.losses.adv import (
    DiscriminatorFeatureMatchingLoss,
    MultiScaleHingeDiscriminatorLoss,
    MultiScaleHingeGeneratorLoss,
)
from terrain_gan.losses.terrain import (
    StructureFeatureSupervisionLoss,
    StructureFeatureSupervisionLossParts,
    TerrainShapeLoss,
    TerrainShapeLossParts,
)
from terrain_gan.models.spade_terrain import (
    make_terrain_noise_map,
    sample_generator_latent,
    uses_original_fresh_block_noise,
)
from terrain_gan.models.shadow import lab_like_gamut_loss, render_generator_terrain, render_shadowed_terrain
from terrain_gan.utils.metrics import compute_terrain_quality_metrics
from terrain_gan.utils.state_dict import load_compatible_state_dict


@dataclass
class SPADEGANEpochMetrics:
    """Metrics for texture+height SPADE GAN training."""

    generator_loss: float = 0.0
    discriminator_loss: float = 0.0
    height_detail_discriminator_loss: float = 0.0
    adversarial_loss: float = 0.0
    height_detail_adversarial_loss: float = 0.0
    height_detail_feature_matching_loss: float = 0.0
    feature_matching_loss: float = 0.0
    height_loss: float = 0.0
    texture_loss: float = 0.0
    height_perceptual_loss: float = 0.0
    texture_perceptual_loss: float = 0.0
    lab_gamut_loss: float = 0.0
    rgb_color_loss: float = 0.0
    structure_detail_scale: float = 1.0
    noise_regularization_loss: float = 0.0
    terrain_shape_loss: float = 0.0
    terrain_detail_loss: float = 0.0
    terrain_branch_morphology_loss: float = 0.0
    terrain_structure_feature_loss: float = 0.0
    terrain_water_flatness_loss: float = 0.0
    terrain_coast_sharpness_loss: float = 0.0
    val_height_l1: float = 0.0
    val_texture_l1: float = 0.0
    val_gradient_l1: float = 0.0
    val_laplacian_l1: float = 0.0
    val_target_detail_energy: float = 0.0
    val_generated_detail_energy: float = 0.0
    val_detail_energy_ratio: float = 0.0
    val_water_flatness: float = 0.0
    val_target_water_flatness: float = 0.0
    val_coast_sharpness: float = 0.0
    val_target_coast_sharpness: float = 0.0
    val_noise_height_l1: float = 0.0
    discriminator_real_confidence: float = 0.0
    augmentation_probability: float = 0.0
    generator_updates: int = 0
    discriminator_updates: int = 0


class SPADEGANTrainer:
    """Trainer for the original-style terrain GAN branch."""

    def __init__(
        self,
        generator: nn.Module,
        shadow_renderer: nn.Module,
        discriminator: nn.Module,
        height_detail_discriminator: nn.Module | None,
        train_loader: DataLoader,
        val_loader: DataLoader,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        height_detail_discriminator_optimizer: torch.optim.Optimizer | None,
        height_loss: nn.Module,
        device: torch.device,
        perceptual_loss: nn.Module | None = None,
        terrain_shape_loss: TerrainShapeLoss | None = None,
        structure_feature_loss: StructureFeatureSupervisionLoss | None = None,
        adversarial_weight: float = 1.0,
        height_detail_adversarial_weight: float = 0.0,
        height_detail_feature_matching_weight: float = 0.0,
        feature_matching_weight: float = 10.0,
        height_weight: float = 2.0,
        texture_weight: float = 1.0,
        height_perceptual_weight: float = 0.0,
        texture_perceptual_weight: float = 0.0,
        lab_gamut_weight: float = 0.0,
        rgb_color_weight: float = 0.0,
        noise_regularization_weight: float = 0.0,
        noise_regularization_start_block: int = 4,
        disable_generator_above: float = 2.0,
        enable_generator_below: float = 1.5,
        max_discriminator_only_steps: int | None = None,
        min_discriminator_steps_per_generator: int = 0,
        generator_warmup_epochs: int = 0,
        reconstruction_warmup_epochs: int = 0,
        warmup_texture_weight: float | None = None,
        structure_detail_warmup_epochs: int = 0,
        structure_detail_ramp_epochs: int = 0,
        dynamic_generator_gating: bool = True,
        reset_generator_gate_each_epoch: bool = False,
        ema_decay: float = 0.0,
        use_ema_for_validation: bool = True,
        discriminator_augmentation: dict[str, Any] | None = None,
        adaptive_discriminator_augmentation: dict[str, Any] | None = None,
        amp: bool = False,
        max_train_steps: int | None = None,
        max_val_steps: int | None = None,
        clip_grad_norm: float | None = None,
        height_domain: str = "zero_one",
        channels_last: bool = False,
    ) -> None:
        self.generator = generator
        self.shadow_renderer = shadow_renderer
        self.discriminator = discriminator
        self.height_detail_discriminator = height_detail_discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.height_detail_discriminator_optimizer = height_detail_discriminator_optimizer
        self.height_loss = height_loss
        self.perceptual_loss = perceptual_loss
        self.terrain_shape_loss = terrain_shape_loss
        self.structure_feature_loss = structure_feature_loss
        self.device = device
        self.adversarial_weight = adversarial_weight
        self.height_detail_adversarial_weight = height_detail_adversarial_weight
        self.height_detail_feature_matching_weight = height_detail_feature_matching_weight
        self.feature_matching_weight = feature_matching_weight
        self.height_weight = height_weight
        self.texture_weight = texture_weight
        self.height_perceptual_weight = height_perceptual_weight
        self.texture_perceptual_weight = texture_perceptual_weight
        self.lab_gamut_weight = max(float(lab_gamut_weight), 0.0)
        self.rgb_color_weight = max(float(rgb_color_weight), 0.0)
        self.noise_regularization_weight = max(float(noise_regularization_weight), 0.0)
        self.noise_regularization_start_block = max(int(noise_regularization_start_block), 0)
        self.disable_generator_above = disable_generator_above
        self.enable_generator_below = enable_generator_below
        self.max_discriminator_only_steps = max_discriminator_only_steps
        self.min_discriminator_steps_per_generator = max(int(min_discriminator_steps_per_generator), 0)
        self.generator_warmup_epochs = max(int(generator_warmup_epochs), 0)
        self.reconstruction_warmup_epochs = max(int(reconstruction_warmup_epochs), 0)
        self.warmup_texture_weight = warmup_texture_weight
        self.structure_detail_warmup_epochs = max(int(structure_detail_warmup_epochs), 0)
        self.structure_detail_ramp_epochs = max(int(structure_detail_ramp_epochs), 0)
        self.dynamic_generator_gating = dynamic_generator_gating
        self.reset_generator_gate_each_epoch = reset_generator_gate_each_epoch
        self.ema_decay = float(ema_decay)
        self.use_ema_for_validation = bool(use_ema_for_validation)
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        discriminator_augmentation = discriminator_augmentation or {}
        self.discriminator_augmentation_enabled = bool(discriminator_augmentation.get("enabled", False))
        self.discriminator_augmentation_probability = float(discriminator_augmentation.get("probability", 0.0))
        self.discriminator_augmentation_translate = int(discriminator_augmentation.get("translate", 0))
        self.discriminator_augmentation_cutout_ratio = float(discriminator_augmentation.get("cutout_ratio", 0.0))
        self.discriminator_augmentation_noise_std = float(discriminator_augmentation.get("noise_std", 0.0))
        adaptive_discriminator_augmentation = adaptive_discriminator_augmentation or {}
        self.adaptive_discriminator_augmentation_enabled = bool(
            adaptive_discriminator_augmentation.get("enabled", False)
        )
        if self.adaptive_discriminator_augmentation_enabled:
            self.discriminator_augmentation_enabled = True
            self.discriminator_augmentation_probability = float(
                adaptive_discriminator_augmentation.get(
                    "initial_probability",
                    self.discriminator_augmentation_probability,
                )
            )
            self.discriminator_augmentation_translate = int(
                adaptive_discriminator_augmentation.get("translate", self.discriminator_augmentation_translate)
            )
            self.discriminator_augmentation_cutout_ratio = float(
                adaptive_discriminator_augmentation.get("cutout_ratio", self.discriminator_augmentation_cutout_ratio)
            )
            self.discriminator_augmentation_noise_std = float(
                adaptive_discriminator_augmentation.get("noise_std", self.discriminator_augmentation_noise_std)
            )
        self.augmentation_min_probability = float(adaptive_discriminator_augmentation.get("min_probability", 0.0))
        self.augmentation_max_probability = float(adaptive_discriminator_augmentation.get("max_probability", 1.0))
        self.augmentation_target_real_confidence = float(
            adaptive_discriminator_augmentation.get("target_real_confidence", 0.6)
        )
        self.augmentation_adjustment_speed = float(adaptive_discriminator_augmentation.get("adjustment_speed", 0.05))
        self.augmentation_update_interval_epochs = max(
            int(adaptive_discriminator_augmentation.get("update_interval_epochs", 1)),
            1,
        )
        self.last_discriminator_real_confidence = 0.0
        self.amp = amp and device.type == "cuda"
        self.max_train_steps = max_train_steps
        self.max_val_steps = max_val_steps
        self.clip_grad_norm = clip_grad_norm
        if height_domain not in {"zero_one", "signed"}:
            raise ValueError(f"Unsupported height_domain: {height_domain}")
        self.height_domain = height_domain
        self.height_value_range = "minus_one_one" if height_domain == "signed" else "zero_one"
        self.channels_last = channels_last and device.type == "cuda"
        self.discriminator_loss = MultiScaleHingeDiscriminatorLoss()
        self.generator_loss = MultiScaleHingeGeneratorLoss()
        self.feature_matching_loss = DiscriminatorFeatureMatchingLoss()
        self.generator_scaler = torch.amp.GradScaler(device.type, enabled=self.amp)
        self.discriminator_scaler = torch.amp.GradScaler(device.type, enabled=self.amp)
        self.height_detail_discriminator_scaler = torch.amp.GradScaler(device.type, enabled=self.amp)
        self.global_step = 0
        self.train_generator_enabled = True
        self.discriminator_only_steps = 0

        self.generator.to(self.device)
        self.shadow_renderer.to(self.device)
        self.discriminator.to(self.device)
        if self.height_detail_discriminator is not None:
            self.height_detail_discriminator.to(self.device)
        if self.channels_last:
            self.generator.to(memory_format=torch.channels_last)
            self.shadow_renderer.to(memory_format=torch.channels_last)
            self.discriminator.to(memory_format=torch.channels_last)
            if self.height_detail_discriminator is not None:
                self.height_detail_discriminator.to(memory_format=torch.channels_last)
        if self.perceptual_loss is not None:
            self.perceptual_loss.to(self.device)
            self.perceptual_loss.eval()
        self.ema_generator: nn.Module | None = None
        self.ema_shadow_renderer: nn.Module | None = None
        if self.ema_decay > 0.0:
            self.ema_generator = copy.deepcopy(self.generator).eval()
            self.ema_shadow_renderer = copy.deepcopy(self.shadow_renderer).eval()
            for parameter in self.ema_generator.parameters():
                parameter.requires_grad_(False)
            for parameter in self.ema_shadow_renderer.parameters():
                parameter.requires_grad_(False)

    def state_dict(self) -> dict[str, Any]:
        """Return trainer-only state that is not owned by modules or optimizers."""
        state = {
            "generator_scaler": self.generator_scaler.state_dict(),
            "discriminator_scaler": self.discriminator_scaler.state_dict(),
            "height_detail_discriminator_scaler": self.height_detail_discriminator_scaler.state_dict(),
            "global_step": self.global_step,
            "train_generator_enabled": self.train_generator_enabled,
            "discriminator_only_steps": self.discriminator_only_steps,
            "discriminator_augmentation_probability": self.discriminator_augmentation_probability,
        }
        if self.ema_generator is not None and self.ema_shadow_renderer is not None:
            state["ema_generator"] = self.ema_generator.state_dict()
            state["ema_shadow_renderer"] = self.ema_shadow_renderer.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore trainer-only state from a checkpoint payload."""
        if "generator_scaler" in state:
            _load_grad_scaler_state(self.generator_scaler, state["generator_scaler"])
        if "discriminator_scaler" in state:
            _load_grad_scaler_state(self.discriminator_scaler, state["discriminator_scaler"])
        if "height_detail_discriminator_scaler" in state:
            _load_grad_scaler_state(
                self.height_detail_discriminator_scaler,
                state["height_detail_discriminator_scaler"],
            )
        self.global_step = int(state.get("global_step", self.global_step))
        self.train_generator_enabled = bool(state.get("train_generator_enabled", self.train_generator_enabled))
        self.discriminator_only_steps = int(state.get("discriminator_only_steps", self.discriminator_only_steps))
        self.discriminator_augmentation_probability = float(
            state.get("discriminator_augmentation_probability", self.discriminator_augmentation_probability)
        )
        if self.ema_generator is not None and "ema_generator" in state:
            load_compatible_state_dict(self.ema_generator, state["ema_generator"])
        if self.ema_shadow_renderer is not None and "ema_shadow_renderer" in state:
            load_compatible_state_dict(self.ema_shadow_renderer, state["ema_shadow_renderer"])

    def apply_phase_overrides(
        self,
        loss_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
    ) -> None:
        """Apply per-epoch curriculum settings without changing model topology."""
        if loss_config:
            self._apply_phase_loss_overrides(loss_config)
        if training_config:
            self._apply_phase_training_overrides(training_config)

    def _apply_phase_loss_overrides(self, loss_config: dict[str, Any]) -> None:
        trainer_loss_attrs = {
            "adversarial": "adversarial_weight",
            "height_detail_adversarial": "height_detail_adversarial_weight",
            "height_detail_feature_matching": "height_detail_feature_matching_weight",
            "feature_matching": "feature_matching_weight",
            "height": "height_weight",
            "texture": "texture_weight",
            "height_perceptual": "height_perceptual_weight",
            "texture_perceptual": "texture_perceptual_weight",
            "lab_gamut": "lab_gamut_weight",
            "rgb_color": "rgb_color_weight",
            "noise_regularization": "noise_regularization_weight",
        }
        for key, attr in trainer_loss_attrs.items():
            if key in loss_config:
                setattr(self, attr, max(float(loss_config[key]), 0.0))

        if "noise_regularization_start_block" in loss_config:
            self.noise_regularization_start_block = max(int(loss_config["noise_regularization_start_block"]), 0)

        height_loss_attrs = {
            "l1_height": "l1_weight",
            "gradient_height": "gradient_weight",
            "laplacian_height": "laplacian_weight",
            "smoothness": "smoothness_weight",
            "low_frequency_height": "low_frequency_weight",
            "frequency_height": "frequency_weight",
        }
        for key, attr in height_loss_attrs.items():
            if key in loss_config and hasattr(self.height_loss, attr):
                setattr(self.height_loss, attr, max(float(loss_config[key]), 0.0))

        if self.terrain_shape_loss is None:
            terrain_loss_attrs = {}
        else:
            terrain_loss_attrs = {
                "terrain_detail_energy": "detail_energy_weight",
                "terrain_branch_morphology": "branch_morphology_weight",
                "terrain_water_flatness": "water_flatness_weight",
                "terrain_coast_sharpness": "coast_sharpness_weight",
                "terrain_detail_energy_moment_weight": "detail_energy_moment_weight",
                "terrain_branch_morphology_coherence_weight": "branch_morphology_coherence_weight",
            }
        for key, attr in terrain_loss_attrs.items():
            if key in loss_config and self.terrain_shape_loss is not None:
                setattr(self.terrain_shape_loss, attr, max(float(loss_config[key]), 0.0))

        structure_feature_loss = getattr(self, "structure_feature_loss", None)
        if structure_feature_loss is not None and "terrain_structure_feature_supervision" in loss_config:
            structure_feature_loss.weight = max(
                float(loss_config["terrain_structure_feature_supervision"]),
                0.0,
            )

    def _apply_phase_training_overrides(self, training_config: dict[str, Any]) -> None:
        if "disable_generator_above" in training_config:
            self.disable_generator_above = float(training_config["disable_generator_above"])
        if "enable_generator_below" in training_config:
            self.enable_generator_below = float(training_config["enable_generator_below"])
        if "min_discriminator_steps_per_generator" in training_config:
            self.min_discriminator_steps_per_generator = max(
                int(training_config["min_discriminator_steps_per_generator"]),
                0,
            )
        if "max_discriminator_only_steps" in training_config:
            value = training_config["max_discriminator_only_steps"]
            self.max_discriminator_only_steps = None if value is None else max(int(value), 0)

    def train_one_epoch(self, epoch: int) -> SPADEGANEpochMetrics:
        self.generator.train()
        self.shadow_renderer.train()
        self.discriminator.train()
        if self.height_detail_discriminator is not None:
            self.height_detail_discriminator.train()
        structure_detail_scale = self._apply_structure_detail_schedule(epoch)
        totals = _Totals()
        in_generator_warmup = epoch < self.generator_warmup_epochs
        in_reconstruction_warmup = epoch < self.reconstruction_warmup_epochs
        if self.reset_generator_gate_each_epoch:
            self.train_generator_enabled = True

        for batch in self.train_loader:
            batch = _move_batch(batch, self.device, channels_last=self.channels_last)
            _validate_texture_batch(batch)

            generator_enabled_for_batch = self.train_generator_enabled
            if in_reconstruction_warmup or in_generator_warmup or not self.dynamic_generator_gating:
                generator_enabled_for_batch = True

            d_loss: torch.Tensor | None = None
            if not in_reconstruction_warmup:
                d_loss = self._train_discriminator(batch)
                totals.discriminator_loss += float(d_loss.detach())
                totals.discriminator_real_confidence += self.last_discriminator_real_confidence
                totals.discriminator_updates += 1
                if self.height_detail_discriminator is not None:
                    height_d_loss = self._train_height_detail_discriminator(batch)
                    totals.height_detail_discriminator_loss += float(height_d_loss.detach())

            force_generator_step = in_reconstruction_warmup or in_generator_warmup or (
                self.max_discriminator_only_steps is not None
                and self.discriminator_only_steps >= self.max_discriminator_only_steps
            )
            needs_more_discriminator_steps = (
                not in_reconstruction_warmup
                and not in_generator_warmup
                and not force_generator_step
                and self.discriminator_only_steps < self.min_discriminator_steps_per_generator
            )

            if (generator_enabled_for_batch or force_generator_step) and not needs_more_discriminator_steps:
                g_parts = self._train_generator(
                    batch,
                    use_adversarial=not in_reconstruction_warmup,
                    texture_weight=self.warmup_texture_weight if in_reconstruction_warmup else None,
                )
                totals.generator_loss += float(g_parts.total.detach())
                totals.adversarial_loss += float(g_parts.adversarial.detach())
                totals.height_detail_adversarial_loss += float(g_parts.height_detail_adversarial.detach())
                totals.height_detail_feature_matching_loss += float(
                    g_parts.height_detail_feature_matching.detach()
                )
                totals.feature_matching_loss += float(g_parts.feature_matching.detach())
                totals.height_loss += float(g_parts.height.detach())
                totals.texture_loss += float(g_parts.texture.detach())
                totals.height_perceptual_loss += float(g_parts.height_perceptual.detach())
                totals.texture_perceptual_loss += float(g_parts.texture_perceptual.detach())
                totals.lab_gamut_loss += float(g_parts.lab_gamut.detach())
                totals.rgb_color_loss += float(g_parts.rgb_color.detach())
                totals.noise_regularization_loss += float(g_parts.noise_regularization.detach())
                totals.terrain_shape_loss += float(g_parts.terrain_shape.detach())
                totals.terrain_detail_loss += float(g_parts.terrain_detail.detach())
                totals.terrain_branch_morphology_loss += float(g_parts.terrain_branch_morphology.detach())
                totals.terrain_structure_feature_loss += float(g_parts.terrain_structure_feature.detach())
                totals.terrain_water_flatness_loss += float(g_parts.terrain_water_flatness.detach())
                totals.terrain_coast_sharpness_loss += float(g_parts.terrain_coast_sharpness.detach())
                totals.generator_updates += 1
                self.discriminator_only_steps = 0
                self._update_ema()
            else:
                self.discriminator_only_steps += 1

            if in_reconstruction_warmup or in_generator_warmup:
                self.train_generator_enabled = True
            elif self.dynamic_generator_gating:
                if d_loss is None:
                    raise RuntimeError("discriminator loss is required outside reconstruction warmup")
                if float(d_loss.detach()) >= self.disable_generator_above:
                    self.train_generator_enabled = False
                elif float(d_loss.detach()) < self.enable_generator_below:
                    self.train_generator_enabled = True
            else:
                self.train_generator_enabled = True

            totals.steps += 1
            self.global_step += 1
            if self.max_train_steps is not None and totals.steps >= self.max_train_steps:
                break

        metrics = totals.to_metrics()
        metrics.structure_detail_scale = structure_detail_scale
        self._update_adaptive_discriminator_augmentation(epoch, metrics.discriminator_real_confidence)
        metrics.augmentation_probability = self.discriminator_augmentation_probability
        return metrics

    def validate(self, epoch: int) -> SPADEGANEpochMetrics:
        eval_generator = self.ema_generator if self.use_ema_for_validation and self.ema_generator is not None else self.generator
        eval_shadow_renderer = (
            self.ema_shadow_renderer
            if self.use_ema_for_validation and self.ema_shadow_renderer is not None
            else self.shadow_renderer
        )
        eval_generator.eval()
        eval_shadow_renderer.eval()
        structure_detail_scale = self._structure_detail_scale_for_epoch(epoch)
        self._set_structure_detail_scale(eval_generator, structure_detail_scale)
        if self.height_detail_discriminator is not None:
            self.height_detail_discriminator.eval()
        totals = _Totals()
        noise_generator = _make_torch_generator(self.device, seed=50_000)

        with torch.no_grad():
            for batch in self.val_loader:
                batch = _move_batch(batch, self.device, channels_last=self.channels_last)
                _validate_texture_batch(batch)
                target_height = self._target_height(batch.height)
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    generated, shadowed = render_generator_terrain(
                        eval_generator,
                        eval_shadow_renderer,
                        batch.biome_onehot,
                    )
                    noise_height = self._generate_noise_height(
                        eval_generator,
                        eval_shadow_renderer,
                        batch.biome_onehot,
                        noise_generator,
                    )
                generated_height = shadowed[:, 3:4]
                totals.val_height_l1 += float(F.l1_loss(generated_height, target_height).detach())
                totals.val_texture_l1 += float(F.l1_loss(shadowed[:, :3], batch.texture).detach())
                quality = compute_terrain_quality_metrics(
                    generated_height,
                    target_height,
                    batch.biome_onehot,
                    noise_height=noise_height,
                )
                totals.val_gradient_l1 += quality.gradient_l1
                totals.val_laplacian_l1 += quality.laplacian_l1
                totals.val_target_detail_energy += quality.target_detail_energy
                totals.val_generated_detail_energy += quality.generated_detail_energy
                totals.val_detail_energy_ratio += quality.detail_energy_ratio
                totals.val_water_flatness += quality.water_flatness
                totals.val_target_water_flatness += quality.target_water_flatness
                totals.val_coast_sharpness += quality.coast_sharpness
                totals.val_target_coast_sharpness += quality.target_coast_sharpness
                totals.val_noise_height_l1 += quality.noise_height_l1
                totals.steps += 1
                if self.max_val_steps is not None and totals.steps >= self.max_val_steps:
                    break

        metrics = totals.to_metrics()
        metrics.structure_detail_scale = structure_detail_scale
        return metrics

    def _generate_noise_height(
        self,
        generator: nn.Module,
        shadow_renderer: nn.Module,
        biome_onehot: torch.Tensor,
        noise_generator: torch.Generator,
    ) -> torch.Tensor:
        noise_dim = getattr(generator, "noise_dim", None)
        if noise_dim is None:
            return shadow_renderer(generator(biome_onehot))[:, 3:4]
        noise = sample_generator_latent(
            generator,
            biome_onehot.shape[0],
            biome_onehot.device,
            biome_onehot.dtype,
            rng=noise_generator,
        )
        noise_map = _make_noise_map_for_generator(generator, biome_onehot, noise_generator)
        if uses_original_fresh_block_noise(generator, noise_map):
            seed = _sample_seed_from_generator(noise_generator, biome_onehot.device)
            return shadow_renderer(_forward_with_seeded_internal_noise(generator, biome_onehot, noise, seed))[:, 3:4]
        return shadow_renderer(generator(biome_onehot, noise=noise, noise_map=noise_map))[:, 3:4]

    def _train_discriminator(self, batch: TerrainBatch) -> torch.Tensor:
        _set_requires_grad(self.discriminator, True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)

        target_height = self._target_height(batch.height)
        real_terrain = torch.cat([batch.texture, target_height], dim=1)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.amp):
            _, generated_shadowed = render_generator_terrain(
                self.generator,
                self.shadow_renderer,
                batch.biome_onehot,
            )
            flipped_shadowed, flipped_condition = _random_flip_pair(generated_shadowed, batch.biome_onehot)

        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            real_condition, real_terrain_aug = self._augment_discriminator_inputs(batch.biome_onehot, real_terrain)
            fake_condition, fake_terrain_aug = self._augment_discriminator_inputs(
                batch.biome_onehot,
                generated_shadowed.detach(),
            )
            flipped_condition_aug, flipped_terrain_aug = self._augment_discriminator_inputs(
                flipped_condition,
                flipped_shadowed.detach(),
            )
            real_outputs = self.discriminator(real_condition, real_terrain_aug)
            fake_outputs = self.discriminator(fake_condition, fake_terrain_aug)
            flipped_outputs = self.discriminator(flipped_condition_aug, flipped_terrain_aug)
            self.last_discriminator_real_confidence = float(_real_confidence(real_outputs).detach())
            loss = self.discriminator_loss(real_outputs, fake_outputs) + self.discriminator_loss(
                real_outputs,
                flipped_outputs,
            )

        self.discriminator_scaler.scale(loss).backward()
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
            self.discriminator_scaler.unscale_(self.discriminator_optimizer)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad_norm)
        self.discriminator_scaler.step(self.discriminator_optimizer)
        self.discriminator_scaler.update()
        return loss

    def _train_height_detail_discriminator(self, batch: TerrainBatch) -> torch.Tensor:
        if self.height_detail_discriminator is None or self.height_detail_discriminator_optimizer is None:
            return batch.height.new_tensor(0.0)
        _set_requires_grad(self.height_detail_discriminator, True)
        self.height_detail_discriminator_optimizer.zero_grad(set_to_none=True)

        target_height = self._target_height(batch.height)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.amp):
            _, generated_shadowed = render_generator_terrain(
                self.generator,
                self.shadow_renderer,
                batch.biome_onehot,
            )

        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            real_outputs = self.height_detail_discriminator(batch.biome_onehot, target_height)
            fake_outputs = self.height_detail_discriminator(batch.biome_onehot, generated_shadowed[:, 3:4].detach())
            loss = self.discriminator_loss(real_outputs, fake_outputs)

        self.height_detail_discriminator_scaler.scale(loss).backward()
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
            self.height_detail_discriminator_scaler.unscale_(self.height_detail_discriminator_optimizer)
            torch.nn.utils.clip_grad_norm_(self.height_detail_discriminator.parameters(), self.clip_grad_norm)
        self.height_detail_discriminator_scaler.step(self.height_detail_discriminator_optimizer)
        self.height_detail_discriminator_scaler.update()
        return loss

    def _train_generator(
        self,
        batch: TerrainBatch,
        use_adversarial: bool = True,
        texture_weight: float | None = None,
    ) -> "_GeneratorLossParts":
        _set_requires_grad(self.discriminator, False)
        if self.height_detail_discriminator is not None:
            _set_requires_grad(self.height_detail_discriminator, False)
        self.generator_optimizer.zero_grad(set_to_none=True)
        target_height = self._target_height(batch.height)

        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            generated, generated_shadowed, structure_maps = self._render_generator_for_training(batch.biome_onehot)
            generated_height = generated_shadowed[:, 3:4]
            zero = generated.new_tensor(0.0)
            adversarial = zero
            feature_matching = zero
            height_detail_adversarial = zero
            height_detail_feature_matching = zero

            if use_adversarial:
                real_terrain = torch.cat([batch.texture, target_height], dim=1)
                flipped_shadowed, flipped_condition = _random_flip_pair(generated_shadowed, batch.biome_onehot)

                real_condition, real_terrain_aug = self._augment_discriminator_inputs(batch.biome_onehot, real_terrain)
                fake_condition, fake_terrain_aug = self._augment_discriminator_inputs(
                    batch.biome_onehot,
                    generated_shadowed,
                )
                flipped_condition_aug, flipped_terrain_aug = self._augment_discriminator_inputs(
                    flipped_condition,
                    flipped_shadowed,
                )
                real_outputs = self.discriminator(real_condition, real_terrain_aug)
                fake_outputs = self.discriminator(fake_condition, fake_terrain_aug)
                flipped_outputs = self.discriminator(flipped_condition_aug, flipped_terrain_aug)

                adversarial = self.generator_loss(fake_outputs) + self.generator_loss(flipped_outputs)
                feature_matching = 0.5 * (
                    self.feature_matching_loss(real_outputs, fake_outputs)
                    + self.feature_matching_loss(real_outputs, flipped_outputs)
                )
                height_detail_adversarial, height_detail_feature_matching = self._height_detail_generator_loss(
                    batch.biome_onehot,
                    generated_height,
                    target_height,
                )
            height = self.height_loss(generated_height, target_height)
            texture = F.l1_loss(generated_shadowed[:, :3], batch.texture)
            height_perceptual = self._perceptual_loss(
                generated_height,
                target_height,
                value_range=self.height_value_range,
                weight=self.height_perceptual_weight,
            )
            texture_perceptual = self._perceptual_loss(
                generated_shadowed[:, :3],
                batch.texture,
                value_range="minus_one_one",
                weight=self.texture_perceptual_weight,
            )
            lab_gamut = self._lab_gamut_loss(generated[:, :3])
            rgb_color = self._rgb_color_loss(generated_shadowed[:, :3], batch.texture)
            noise_regularization = self._noise_regularization_loss(generated)
            terrain_shape = self._terrain_shape_loss(generated_height, target_height, batch.biome_onehot)
            terrain_structure_feature = self._terrain_structure_feature_loss(
                structure_maps,
                target_height,
                batch.biome_onehot,
                generated_height,
            )
            active_texture_weight = self.texture_weight if texture_weight is None else float(texture_weight)
            total = (
                self.adversarial_weight * adversarial
                + self.height_detail_adversarial_weight * height_detail_adversarial
                + self.height_detail_feature_matching_weight * height_detail_feature_matching
                + self.feature_matching_weight * feature_matching
                + self.height_weight * height
                + active_texture_weight * texture
                + self.height_perceptual_weight * height_perceptual
                + self.texture_perceptual_weight * texture_perceptual
                + self.lab_gamut_weight * lab_gamut
                + self.rgb_color_weight * rgb_color
                + self.noise_regularization_weight * noise_regularization
                + terrain_shape.total
                + terrain_structure_feature.total
            )

        self.generator_scaler.scale(total).backward()
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
            self.generator_scaler.unscale_(self.generator_optimizer)
            params = list(self.generator.parameters()) + list(self.shadow_renderer.parameters())
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.generator_scaler.step(self.generator_optimizer)
        self.generator_scaler.update()
        _set_requires_grad(self.discriminator, True)
        if self.height_detail_discriminator is not None:
            _set_requires_grad(self.height_detail_discriminator, True)
        return _GeneratorLossParts(
            total,
            adversarial,
            height_detail_adversarial,
            height_detail_feature_matching,
            feature_matching,
            height,
            texture,
            height_perceptual,
            texture_perceptual,
            lab_gamut,
            rgb_color,
            noise_regularization,
            terrain_shape.total,
            terrain_shape.detail_energy,
            terrain_shape.branch_morphology,
            terrain_structure_feature.total,
            terrain_shape.water_flatness,
            terrain_shape.coast_sharpness,
        )

    def _render_generator_for_training(
        self,
        biome_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.structure_feature_loss is not None and self.structure_feature_loss.enabled:
            forward_with_structure = getattr(self.generator, "forward_with_base_output_and_structure", None)
            if not callable(forward_with_structure):
                raise RuntimeError(
                    "terrain_structure_feature_supervision is enabled, but the generator does not expose "
                    "forward_with_base_output_and_structure()."
                )
            generated, base_generated, structure_maps = forward_with_structure(biome_onehot, noise=None, noise_map=None)
            if structure_maps is None:
                raise RuntimeError(
                    "terrain_structure_feature_supervision is enabled, but the generator did not return "
                    "StructureCarver maps."
                )
            return generated, render_shadowed_terrain(self.shadow_renderer, generated, base_generated), structure_maps

        generated, generated_shadowed = render_generator_terrain(
            self.generator,
            self.shadow_renderer,
            biome_onehot,
        )
        return generated, generated_shadowed, None

    def _augment_discriminator_inputs(
        self,
        condition: torch.Tensor,
        terrain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.discriminator_augmentation_enabled:
            return condition, terrain
        if self.discriminator_augmentation_probability <= 0.0:
            return condition, terrain
        if torch.rand((), device=terrain.device) >= self.discriminator_augmentation_probability:
            return condition, terrain
        if self.discriminator_augmentation_translate > 0:
            condition, terrain = _random_translate_pair(condition, terrain, self.discriminator_augmentation_translate)
        if self.discriminator_augmentation_noise_std > 0.0:
            terrain = (terrain + torch.randn_like(terrain) * self.discriminator_augmentation_noise_std).clamp(-1.0, 1.0)
        if self.discriminator_augmentation_cutout_ratio > 0.0:
            terrain = _random_cutout_terrain(terrain, self.discriminator_augmentation_cutout_ratio)
        return condition, terrain

    def _update_adaptive_discriminator_augmentation(self, epoch: int, real_confidence: float) -> None:
        if not self.adaptive_discriminator_augmentation_enabled:
            return
        if epoch % self.augmentation_update_interval_epochs != 0:
            return
        delta = (real_confidence - self.augmentation_target_real_confidence) * self.augmentation_adjustment_speed
        updated = self.discriminator_augmentation_probability + delta
        self.discriminator_augmentation_probability = min(
            self.augmentation_max_probability,
            max(self.augmentation_min_probability, updated),
        )

    def _update_ema(self) -> None:
        if self.ema_generator is None or self.ema_shadow_renderer is None:
            return
        _ema_update(self.ema_generator, self.generator, self.ema_decay)
        _ema_update(self.ema_shadow_renderer, self.shadow_renderer, self.ema_decay)

    def _apply_structure_detail_schedule(self, epoch: int) -> float:
        scale = self._structure_detail_scale_for_epoch(epoch)
        self._set_structure_detail_scale(self.generator, scale)
        if self.ema_generator is not None:
            self._set_structure_detail_scale(self.ema_generator, scale)
        return scale

    def _structure_detail_scale_for_epoch(self, epoch: int) -> float:
        if self.structure_detail_warmup_epochs <= 0 and self.structure_detail_ramp_epochs <= 0:
            return 1.0
        if epoch < self.structure_detail_warmup_epochs:
            return 0.0
        if self.structure_detail_ramp_epochs <= 0:
            return 1.0
        ramp_epoch = epoch - self.structure_detail_warmup_epochs + 1
        return min(1.0, max(0.0, ramp_epoch / self.structure_detail_ramp_epochs))

    @staticmethod
    def _set_structure_detail_scale(module: nn.Module, scale: float) -> None:
        setter = getattr(module, "set_structure_detail_runtime_scale", None)
        if callable(setter):
            setter(scale)
            return
        wrapped = getattr(module, "module", None)
        setter = getattr(wrapped, "set_structure_detail_runtime_scale", None)
        if callable(setter):
            setter(scale)

    def _height_detail_generator_loss(
        self,
        biome_onehot: torch.Tensor,
        generated_height: torch.Tensor,
        target_height: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.height_detail_discriminator is None:
            zero = generated_height.new_tensor(0.0)
            return zero, zero
        real_outputs = self.height_detail_discriminator(biome_onehot, target_height)
        fake_outputs = self.height_detail_discriminator(biome_onehot, generated_height)
        adversarial = self.generator_loss(fake_outputs)
        feature_matching = self.feature_matching_loss(real_outputs, fake_outputs)
        return adversarial, feature_matching

    def _perceptual_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        value_range: str,
        weight: float,
    ) -> torch.Tensor:
        if self.perceptual_loss is None or weight <= 0.0:
            return prediction.new_tensor(0.0)
        return self.perceptual_loss(prediction, target, value_range=value_range)

    def _lab_gamut_loss(self, raw_texture_lab_like: torch.Tensor) -> torch.Tensor:
        if self.lab_gamut_weight <= 0.0:
            return raw_texture_lab_like.new_tensor(0.0)
        return lab_like_gamut_loss(torch.tanh(raw_texture_lab_like))

    def _rgb_color_loss(self, generated_texture: torch.Tensor, target_texture: torch.Tensor) -> torch.Tensor:
        if self.rgb_color_weight <= 0.0:
            return generated_texture.new_tensor(0.0)
        generated_mean = generated_texture.mean(dim=(2, 3))
        target_mean = target_texture.mean(dim=(2, 3))
        generated_std = generated_texture.std(dim=(2, 3), unbiased=False)
        target_std = target_texture.std(dim=(2, 3), unbiased=False)
        height, width = generated_texture.shape[-2:]
        pooling = max(1, min(16, height, width))
        generated_broad = F.avg_pool2d(generated_texture, kernel_size=pooling, stride=pooling)
        target_broad = F.avg_pool2d(target_texture, kernel_size=pooling, stride=pooling)
        return (
            F.l1_loss(generated_mean, target_mean)
            + 0.5 * F.l1_loss(generated_std, target_std)
            + 0.5 * F.l1_loss(generated_broad, target_broad)
        )

    def _noise_regularization_loss(self, reference: torch.Tensor) -> torch.Tensor:
        if self.noise_regularization_weight <= 0.0:
            return reference.new_tensor(0.0)
        penalties: list[torch.Tensor] = []
        for name, parameter in self.generator.named_parameters():
            block_index = _noise_injection_block_index(name)
            if block_index is None or block_index < self.noise_regularization_start_block:
                continue
            penalties.append(parameter.float().square().mean())
        if not penalties:
            return reference.new_tensor(0.0)
        return torch.stack(penalties).mean().to(device=reference.device)

    def _terrain_shape_loss(
        self,
        generated_height: torch.Tensor,
        target_height: torch.Tensor,
        biome_onehot: torch.Tensor,
    ) -> TerrainShapeLossParts:
        if self.terrain_shape_loss is None or not self.terrain_shape_loss.enabled:
            zero = generated_height.new_tensor(0.0)
            return TerrainShapeLossParts(
                total=zero,
                detail_energy=zero,
                branch_morphology=zero,
                water_flatness=zero,
                coast_sharpness=zero,
            )
        return self.terrain_shape_loss(generated_height, target_height, biome_onehot)

    def _terrain_structure_feature_loss(
        self,
        structure_maps: torch.Tensor | None,
        target_height: torch.Tensor,
        biome_onehot: torch.Tensor,
        reference: torch.Tensor,
    ) -> StructureFeatureSupervisionLossParts:
        if self.structure_feature_loss is None or not self.structure_feature_loss.enabled:
            zero = reference.new_tensor(0.0)
            return StructureFeatureSupervisionLossParts(
                total=zero,
                ridge=zero,
                valley=zero,
                drainage=zero,
                detail=zero,
            )
        if structure_maps is None:
            raise RuntimeError("Structure feature supervision requires StructureCarver maps")
        return self.structure_feature_loss(structure_maps, target_height, biome_onehot)

    def _target_height(self, height: torch.Tensor) -> torch.Tensor:
        if self.height_domain == "signed":
            return height * 2.0 - 1.0
        return height


@dataclass
class _GeneratorLossParts:
    total: torch.Tensor
    adversarial: torch.Tensor
    height_detail_adversarial: torch.Tensor
    height_detail_feature_matching: torch.Tensor
    feature_matching: torch.Tensor
    height: torch.Tensor
    texture: torch.Tensor
    height_perceptual: torch.Tensor
    texture_perceptual: torch.Tensor
    lab_gamut: torch.Tensor
    rgb_color: torch.Tensor
    noise_regularization: torch.Tensor
    terrain_shape: torch.Tensor
    terrain_detail: torch.Tensor
    terrain_branch_morphology: torch.Tensor
    terrain_structure_feature: torch.Tensor
    terrain_water_flatness: torch.Tensor
    terrain_coast_sharpness: torch.Tensor


@dataclass
class _Totals:
    generator_loss: float = 0.0
    discriminator_loss: float = 0.0
    height_detail_discriminator_loss: float = 0.0
    adversarial_loss: float = 0.0
    height_detail_adversarial_loss: float = 0.0
    height_detail_feature_matching_loss: float = 0.0
    feature_matching_loss: float = 0.0
    height_loss: float = 0.0
    texture_loss: float = 0.0
    height_perceptual_loss: float = 0.0
    texture_perceptual_loss: float = 0.0
    lab_gamut_loss: float = 0.0
    rgb_color_loss: float = 0.0
    noise_regularization_loss: float = 0.0
    terrain_shape_loss: float = 0.0
    terrain_detail_loss: float = 0.0
    terrain_branch_morphology_loss: float = 0.0
    terrain_structure_feature_loss: float = 0.0
    terrain_water_flatness_loss: float = 0.0
    terrain_coast_sharpness_loss: float = 0.0
    val_height_l1: float = 0.0
    val_texture_l1: float = 0.0
    val_gradient_l1: float = 0.0
    val_laplacian_l1: float = 0.0
    val_target_detail_energy: float = 0.0
    val_generated_detail_energy: float = 0.0
    val_detail_energy_ratio: float = 0.0
    val_water_flatness: float = 0.0
    val_target_water_flatness: float = 0.0
    val_coast_sharpness: float = 0.0
    val_target_coast_sharpness: float = 0.0
    val_noise_height_l1: float = 0.0
    discriminator_real_confidence: float = 0.0
    augmentation_probability: float = 0.0
    steps: int = 0
    generator_updates: int = 0
    discriminator_updates: int = 0

    def to_metrics(self) -> SPADEGANEpochMetrics:
        generator_steps = max(self.generator_updates, 1)
        val_steps = max(self.steps, 1)
        return SPADEGANEpochMetrics(
            generator_loss=self.generator_loss / generator_steps,
            discriminator_loss=self.discriminator_loss / max(self.discriminator_updates, 1),
            height_detail_discriminator_loss=self.height_detail_discriminator_loss
            / max(self.discriminator_updates, 1),
            adversarial_loss=self.adversarial_loss / generator_steps,
            height_detail_adversarial_loss=self.height_detail_adversarial_loss / generator_steps,
            height_detail_feature_matching_loss=self.height_detail_feature_matching_loss / generator_steps,
            feature_matching_loss=self.feature_matching_loss / generator_steps,
            height_loss=self.height_loss / generator_steps,
            texture_loss=self.texture_loss / generator_steps,
            height_perceptual_loss=self.height_perceptual_loss / generator_steps,
            texture_perceptual_loss=self.texture_perceptual_loss / generator_steps,
            lab_gamut_loss=self.lab_gamut_loss / generator_steps,
            rgb_color_loss=self.rgb_color_loss / generator_steps,
            noise_regularization_loss=self.noise_regularization_loss / generator_steps,
            terrain_shape_loss=self.terrain_shape_loss / generator_steps,
            terrain_detail_loss=self.terrain_detail_loss / generator_steps,
            terrain_branch_morphology_loss=self.terrain_branch_morphology_loss / generator_steps,
            terrain_structure_feature_loss=self.terrain_structure_feature_loss / generator_steps,
            terrain_water_flatness_loss=self.terrain_water_flatness_loss / generator_steps,
            terrain_coast_sharpness_loss=self.terrain_coast_sharpness_loss / generator_steps,
            val_height_l1=self.val_height_l1 / val_steps,
            val_texture_l1=self.val_texture_l1 / val_steps,
            val_gradient_l1=self.val_gradient_l1 / val_steps,
            val_laplacian_l1=self.val_laplacian_l1 / val_steps,
            val_target_detail_energy=self.val_target_detail_energy / val_steps,
            val_generated_detail_energy=self.val_generated_detail_energy / val_steps,
            val_detail_energy_ratio=self.val_detail_energy_ratio / val_steps,
            val_water_flatness=self.val_water_flatness / val_steps,
            val_target_water_flatness=self.val_target_water_flatness / val_steps,
            val_coast_sharpness=self.val_coast_sharpness / val_steps,
            val_target_coast_sharpness=self.val_target_coast_sharpness / val_steps,
            val_noise_height_l1=self.val_noise_height_l1 / val_steps,
            discriminator_real_confidence=self.discriminator_real_confidence / max(self.discriminator_updates, 1),
            augmentation_probability=self.augmentation_probability,
            generator_updates=self.generator_updates,
            discriminator_updates=self.discriminator_updates,
        )


def _move_batch(batch: TerrainBatch, device: torch.device, channels_last: bool = False) -> TerrainBatch:
    def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        moved = tensor.to(device, non_blocking=device.type == "cuda")
        if channels_last and moved.ndim == 4:
            moved = moved.contiguous(memory_format=torch.channels_last)
        return moved

    return TerrainBatch(
        sample_id=batch.sample_id,
        biome_onehot=move(batch.biome_onehot),
        height=move(batch.height),
        texture=move(batch.texture),
    )


def _validate_texture_batch(batch: TerrainBatch) -> None:
    if batch.texture is None:
        raise ValueError("SPADE terrain GAN requires texture targets. Use TerrainPairDataset(..., use_texture=True).")


def _random_flip_pair(terrain: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand((), device=terrain.device) < 0.5:
        terrain = torch.flip(terrain, dims=(3,))
        condition = torch.flip(condition, dims=(3,))
    if torch.rand((), device=terrain.device) < 0.5:
        terrain = torch.flip(terrain, dims=(2,))
        condition = torch.flip(condition, dims=(2,))
    return terrain, condition


def _random_translate_pair(
    condition: torch.Tensor,
    terrain: torch.Tensor,
    max_shift: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_shift <= 0:
        return condition, terrain
    height, width = terrain.shape[-2:]
    pad = int(max_shift)
    top = int(torch.randint(0, 2 * pad + 1, (), device=terrain.device).item())
    left = int(torch.randint(0, 2 * pad + 1, (), device=terrain.device).item())
    condition_padded = F.pad(condition, (pad, pad, pad, pad), mode="replicate")
    terrain_padded = F.pad(terrain, (pad, pad, pad, pad), mode="reflect")
    return (
        condition_padded[..., top : top + height, left : left + width],
        terrain_padded[..., top : top + height, left : left + width],
    )


def _random_cutout_terrain(terrain: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio <= 0.0:
        return terrain
    batch, _, height, width = terrain.shape
    cut_height = max(1, min(height, int(round(height * ratio))))
    cut_width = max(1, min(width, int(round(width * ratio))))
    mask = torch.ones(batch, 1, height, width, device=terrain.device, dtype=terrain.dtype)
    max_top = max(height - cut_height, 0)
    max_left = max(width - cut_width, 0)
    for index in range(batch):
        top = int(torch.randint(0, max_top + 1, (), device=terrain.device).item())
        left = int(torch.randint(0, max_left + 1, (), device=terrain.device).item())
        mask[index, :, top : top + cut_height, left : left + cut_width] = 0.0
    return terrain * mask


def _real_confidence(outputs: list) -> torch.Tensor:
    confidences = [torch.sigmoid(output.logits).mean() for output in outputs]
    return torch.stack(confidences).mean()


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _load_grad_scaler_state(scaler: torch.amp.GradScaler, state: dict[str, Any]) -> None:
    if not scaler.is_enabled() or not state:
        return
    scaler.load_state_dict(state)


def _noise_injection_block_index(parameter_name: str) -> int | None:
    parts = parameter_name.split(".")
    if len(parts) != 3:
        return None
    block_name, noise_name, parameter = parts
    if parameter != "weight" or not block_name.startswith("block") or not noise_name.startswith("noise"):
        return None
    suffix = block_name.removeprefix("block")
    if not suffix.isdigit():
        return None
    return int(suffix)


@torch.no_grad()
def _ema_update(ema_module: nn.Module, source_module: nn.Module, decay: float) -> None:
    ema_state = ema_module.state_dict()
    source_state = source_module.state_dict()
    for key, ema_value in ema_state.items():
        source_value = source_state[key].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(
                source_value.to(device=ema_value.device, dtype=ema_value.dtype),
                alpha=1.0 - decay,
            )
        else:
            ema_value.copy_(source_value.to(device=ema_value.device, dtype=ema_value.dtype))


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _make_noise_map_for_generator(
    generator: nn.Module,
    biome_onehot: torch.Tensor,
    noise_generator: torch.Generator,
) -> torch.Tensor | None:
    if uses_original_fresh_block_noise(generator):
        return None
    if getattr(generator, "noise_map_mode", "") == "terrain_fractal":
        return make_terrain_noise_map(
            biome_onehot.shape[0],
            biome_onehot.shape[-2:],
            biome_onehot.device,
            biome_onehot.dtype,
            randomize=True,
            generator=noise_generator,
        )
    return torch.randn(
        biome_onehot.shape[0],
        1,
        biome_onehot.shape[-2],
        biome_onehot.shape[-1],
        device=biome_onehot.device,
        dtype=biome_onehot.dtype,
        generator=noise_generator,
    )


def _sample_seed_from_generator(generator: torch.Generator, device: torch.device) -> int:
    return int(
        torch.randint(
            0,
            2_147_483_647,
            (),
            device=device,
            generator=generator,
        ).item()
    )


def _forward_with_seeded_internal_noise(
    generator: nn.Module,
    biome_onehot: torch.Tensor,
    noise: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    was_training = generator.training
    cuda_devices: list[int] = []
    if biome_onehot.device.type == "cuda":
        cuda_devices = [
            biome_onehot.device.index if biome_onehot.device.index is not None else torch.cuda.current_device()
        ]
    try:
        generator.train()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if biome_onehot.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            return generator(biome_onehot, noise=noise, noise_map=None)
    finally:
        if not was_training:
            generator.eval()
