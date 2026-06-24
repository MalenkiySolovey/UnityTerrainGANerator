"""Train original-style SPADE terrain GAN."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from terrain_gan.data.control_maps import expected_control_channels, parse_control_map_config
from terrain_gan.data.dataset import TerrainPairDataset, collate_terrain_samples
from terrain_gan.data.palette import PaletteEncoder, PaletteMatcher
from terrain_gan.data.transforms import EvalPairedTransform, TrainPairedTransform, TransformConfig
from terrain_gan.losses.height import CombinedHeightLoss
from terrain_gan.losses.perceptual import VGGPerceptualLoss
from terrain_gan.losses.terrain import StructureFeatureSupervisionLoss, TerrainShapeLoss
from terrain_gan.models.factory import build_generator
from terrain_gan.models.generator import GeneratorConfig
from terrain_gan.models.shadow import (
    LearnedShadowRenderer,
    OriginalLabShadowRenderer,
    OriginalLearnedShadowRenderer,
    OriginalLuminanceShadowRenderer,
    render_generator_terrain,
    render_shadowed_terrain,
)
from terrain_gan.models.spade_terrain import (
    make_terrain_noise_map,
    sample_generator_latent,
    uses_internal_original_noise,
    uses_original_fresh_block_noise,
)
from terrain_gan.models.terrain_discriminator import MultiScaleHeightDetailDiscriminator, MultiScaleTerrainDiscriminator
from terrain_gan.training.checkpoint import (
    CheckpointManager,
    CheckpointState,
    capture_rng_state,
    move_optimizer_state_to_device,
    restore_rng_state,
)
from terrain_gan.training.config import load_training_config
from terrain_gan.training.spade_gan import SPADEGANEpochMetrics, SPADEGANTrainer
from terrain_gan.utils.paths import resolve_path
from terrain_gan.utils.repro import SeedConfig, seed_everything
from terrain_gan.utils.state_dict import load_compatible_state_dict
from terrain_gan.visualization.preview import save_terrain_gan_preview, save_terrain_gan_preview_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SPADE terrain GAN on biome, texture, and height triplets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_terrain_gan.yaml"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-generator", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_training_config(args.config)
    raw = config.raw
    data_config = config.section("data")
    model_config = config.section("model")
    shadow_config = config.section("shadow")
    discriminator_config = config.section("discriminator")
    train_config = config.section("training")
    loss_config = config.section("loss")
    preview_config = config.section("preview")

    seed_everything(SeedConfig(seed=int(train_config.get("seed", 42))))
    device = _select_device(args.device)
    _configure_cuda_backend(train_config, device)

    macro_classes = int(data_config.get("macro_classes", 7))
    control_map_config = parse_control_map_config(data_config.get("control_maps"))
    expected_input_channels = macro_classes + expected_control_channels(control_map_config)
    configured_input_channels = int(model_config.get("input_channels", macro_classes))
    if configured_input_channels != expected_input_channels:
        raise ValueError(
            "model.input_channels must match data.macro_classes plus enabled control channels: "
            f"{configured_input_channels} != {expected_input_channels}"
        )

    matcher = PaletteMatcher.from_files(Path("configs/palette.yaml"), Path("configs/biomes.yaml"))
    encoder = PaletteEncoder(matcher, num_macro_classes=macro_classes)
    image_size = int(data_config.get("image_size", 256))
    train_transform = TrainPairedTransform(
        TransformConfig(image_size=image_size, random_flip=False, random_rotate=False, random_crop=True)
    )
    val_transform = EvalPairedTransform(
        TransformConfig(image_size=image_size, random_flip=False, random_rotate=False, random_crop=False)
    )
    train_dataset = TerrainPairDataset(
        root=resolve_path(data_config["root"]),
        split_csv=resolve_path(data_config["train_split"]),
        palette_encoder=encoder,
        transform=train_transform,
        use_texture=True,
        control_map_config=control_map_config,
    )
    val_dataset = TerrainPairDataset(
        root=resolve_path(data_config["root"]),
        split_csv=resolve_path(data_config["val_split"]),
        palette_encoder=encoder,
        transform=val_transform,
        use_texture=True,
        control_map_config=replace(control_map_config, dropout_probability=0.0),
    )

    num_workers = args.num_workers if args.num_workers is not None else int(train_config.get("num_workers", 0))
    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_terrain_samples,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(train_config.get("persistent_workers", False))
    batch_size = int(train_config.get("batch_size", 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs)

    generator = build_generator(
        GeneratorConfig(
            name=str(model_config.get("name", "spade_terrain_generator")),
            input_channels=int(model_config.get("input_channels", 7)),
            output_mode=str(model_config.get("output_mode", "terrain4")),
            base_channels=int(model_config.get("base_channels", 32)),
            noise_dim=int(model_config.get("noise_dim", 16)),
            splat_channels=int(model_config.get("splat_channels", 0)),
            height_activation=str(model_config.get("height_activation", "sigmoid")),
            latent_mode=str(model_config.get("latent_mode", "spatial")),
            semantic_resize_mode=str(model_config.get("semantic_resize_mode", "nearest")),
            semantic_class_channels=int(model_config.get("semantic_class_channels", 0)),
            noise_injection_count=int(model_config.get("noise_injection_count", 2)),
            noise_feature_strength=_optional_float(model_config.get("noise_feature_strength")),
            norm_before_first_conv=_bool_config(model_config.get("norm_before_first_conv", True)),
            norm_before_second_conv=_bool_config(model_config.get("norm_before_second_conv", True)),
            noise_start_block=int(model_config.get("noise_start_block", 0)),
            modulation_mode=str(model_config.get("modulation_mode", "tanh")),
            modulation_strength=float(model_config.get("modulation_strength", 0.5)),
            noise_map_mode=str(model_config.get("noise_map_mode", "terrain_fractal")),
            latent_distribution=str(model_config.get("latent_distribution", "uniform")),
            output_activation_mode=str(model_config.get("output_activation_mode", "activated")),
            height_detail_branch=_bool_config(model_config.get("height_detail_branch", False)),
            height_detail_strength=float(model_config.get("height_detail_strength", 0.0)),
            height_detail_hidden_channels=int(model_config.get("height_detail_hidden_channels", 32)),
            height_detail_seed_channels=int(model_config.get("height_detail_seed_channels", 8)),
            height_detail_seed_size=int(model_config.get("height_detail_seed_size", 8)),
            height_detail_pattern_channels=int(model_config.get("height_detail_pattern_channels", 4)),
            height_detail_highpass_kernel=int(model_config.get("height_detail_highpass_kernel", 9)),
            height_detail_roughness_floor=float(model_config.get("height_detail_roughness_floor", 0.12)),
            height_detail_water_channel=int(model_config.get("height_detail_water_channel", 0)),
            height_detail_water_flatness_channel=int(model_config.get("height_detail_water_flatness_channel", 7)),
            height_detail_roughness_channel=int(model_config.get("height_detail_roughness_channel", 10)),
            height_detail_detach_inputs=_bool_config(model_config.get("height_detail_detach_inputs", False)),
            height_detail_shadow_source=str(model_config.get("height_detail_shadow_source", "final")),
            structure_detail_branch=_bool_config(model_config.get("structure_detail_branch", False)),
            structure_detail_strength=float(model_config.get("structure_detail_strength", 0.12)),
            structure_detail_hidden_channels=int(model_config.get("structure_detail_hidden_channels", 32)),
            structure_detail_seed_channels=int(model_config.get("structure_detail_seed_channels", 8)),
            structure_detail_seed_size=int(model_config.get("structure_detail_seed_size", 16)),
            structure_detail_procedural_mix=float(model_config.get("structure_detail_procedural_mix", 0.35)),
            structure_detail_highpass_kernel=int(model_config.get("structure_detail_highpass_kernel", 9)),
            structure_detail_roughness_floor=float(model_config.get("structure_detail_roughness_floor", 0.18)),
            structure_ridge_lift=float(model_config.get("structure_ridge_lift", 0.10)),
            structure_valley_cut=float(model_config.get("structure_valley_cut", 0.12)),
            structure_drainage_cut=float(model_config.get("structure_drainage_cut", 0.14)),
            structure_micro_detail=float(model_config.get("structure_micro_detail", 0.035)),
            structure_seed_micro_strength=float(model_config.get("structure_seed_micro_strength", 0.025)),
            structure_spectral_prior_strength=float(model_config.get("structure_spectral_prior_strength", 0.0)),
            structure_spectral_prior_ridged_mix=float(model_config.get("structure_spectral_prior_ridged_mix", 0.55)),
            structure_spectral_prior_bandpass_kernel=int(
                model_config.get("structure_spectral_prior_bandpass_kernel", 7)
            ),
            structure_spectral_prior_broad_kernel=int(model_config.get("structure_spectral_prior_broad_kernel", 31)),
            structure_cell_size=int(model_config.get("structure_cell_size", 16)),
            structure_control_mode=str(model_config.get("structure_control_mode", "legacy")),
            structure_height_prior_channel=int(model_config.get("structure_height_prior_channel", 7)),
            structure_detail_budget_channel=int(model_config.get("structure_detail_budget_channel", 8)),
            structure_boundary_distance_channel=int(model_config.get("structure_boundary_distance_channel", 9)),
            structure_semantic_complexity_channel=int(model_config.get("structure_semantic_complexity_channel", 10)),
            structure_control_adapter_channels=int(model_config.get("structure_control_adapter_channels", 16)),
        )
    )
    shadow_renderer = _build_shadow_renderer(shadow_config)
    discriminator = MultiScaleTerrainDiscriminator(
        semantic_channels=int(model_config.get("input_channels", 7)),
        base_channels=int(discriminator_config.get("base_channels", 32)),
        scales=tuple(float(scale) for scale in discriminator_config.get("scales", [1.0, 0.25, 0.0625])),
        use_spectral_norm=bool(discriminator_config.get("use_spectral_norm", True)),
        use_instance_norm=bool(discriminator_config.get("use_instance_norm", True)),
        slope_mode=str(discriminator_config.get("slope_mode", "standard")),
        activation=str(discriminator_config.get("activation", "leaky_relu")),
    )
    height_detail_discriminator = _build_height_detail_discriminator(discriminator_config, model_config)

    generator_params = list(generator.parameters()) + list(shadow_renderer.parameters())
    generator_optimizer = torch.optim.AdamW(
        generator_params,
        lr=float(train_config.get("generator_learning_rate", 1e-4)),
        betas=(float(train_config.get("adam_beta1", 0.0)), float(train_config.get("adam_beta2", 0.99))),
        weight_decay=float(train_config.get("generator_weight_decay", 0.0)),
        fused=_use_fused_adamw(train_config, device),
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(train_config.get("discriminator_learning_rate", 2e-4)),
        betas=(float(train_config.get("adam_beta1", 0.0)), float(train_config.get("adam_beta2", 0.99))),
        weight_decay=float(train_config.get("discriminator_weight_decay", 0.0)),
        fused=_use_fused_adamw(train_config, device),
    )
    height_detail_discriminator_optimizer = None
    if height_detail_discriminator is not None:
        height_detail_discriminator_optimizer = torch.optim.AdamW(
            height_detail_discriminator.parameters(),
            lr=float(
                train_config.get(
                    "height_detail_discriminator_learning_rate",
                    train_config.get("discriminator_learning_rate", 2e-4),
                )
            ),
            betas=(float(train_config.get("adam_beta1", 0.0)), float(train_config.get("adam_beta2", 0.99))),
            weight_decay=float(train_config.get("discriminator_weight_decay", 0.0)),
            fused=_use_fused_adamw(train_config, device),
        )
    height_loss = CombinedHeightLoss(
        l1_weight=float(loss_config.get("l1_height", 0.5)),
        gradient_weight=float(loss_config.get("gradient_height", 0.5)),
        smoothness_weight=float(loss_config.get("smoothness", 0.0)),
        laplacian_weight=float(loss_config.get("laplacian_height", 0.1)),
        low_frequency_weight=float(loss_config.get("low_frequency_height", 0.0)),
        low_frequency_factor=int(loss_config.get("low_frequency_factor", 16)),
        frequency_weight=float(loss_config.get("frequency_height", 0.0)),
        frequency_min=float(loss_config.get("frequency_min", 0.06)),
    )
    perceptual_loss = _build_perceptual_loss(loss_config)
    terrain_shape_loss = _build_terrain_shape_loss(loss_config)
    structure_feature_loss = _build_structure_feature_loss(loss_config)

    checkpoint_dir = args.checkpoint_dir or Path(
        train_config.get("checkpoint_dir", "artifacts/checkpoints/spade_terrain_gan_256")
    )
    preview_dir = args.preview_dir or Path(
        train_config.get("preview_dir", "artifacts/previews/spade_terrain_gan_256")
    )
    metrics_path = resolve_path(
        args.metrics_path
        or Path(train_config.get("metrics_path", "artifacts/reports/spade_terrain_gan_256_metrics.csv"))
    )
    checkpoint_manager = CheckpointManager(resolve_path(checkpoint_dir))

    best_val = float("inf")
    start_epoch = 0
    resumed_global_step = 0
    resume_state: CheckpointState | None = None
    if args.resume is not None and args.init_generator is not None:
        raise ValueError("Use either --resume for the same architecture or --init-generator for partial weight init.")
    if args.init_generator is not None:
        init_state = checkpoint_manager.load(resolve_path(args.init_generator))
        loaded_generator, skipped_generator = _load_compatible_module_state(generator, init_state.payload["model"])
        loaded_shadow = 0
        skipped_shadow = 0
        init_shadow_from_checkpoint = bool(train_config.get("init_shadow_renderer_from_checkpoint", True))
        if init_shadow_from_checkpoint and "shadow_renderer" in init_state.payload:
            loaded_shadow, skipped_shadow = _load_compatible_module_state(
                shadow_renderer,
                init_state.payload["shadow_renderer"],
            )
        print(
            f"initialized_from={resolve_path(args.init_generator)} "
            f"generator_loaded={loaded_generator} generator_skipped={skipped_generator} "
            f"shadow_loaded={loaded_shadow} shadow_skipped={skipped_shadow}"
            f" shadow_init_from_checkpoint={init_shadow_from_checkpoint}"
        )
    if args.resume is not None:
        resume_state = checkpoint_manager.load(resolve_path(args.resume))
        _restore_state(
            resume_state,
            generator=generator,
            shadow_renderer=shadow_renderer,
            discriminator=discriminator,
            height_detail_discriminator=height_detail_discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            height_detail_discriminator_optimizer=height_detail_discriminator_optimizer,
            device=device,
        )
        best_val = resume_state.best_metric if resume_state.best_metric is not None else float("inf")
        if bool(train_config.get("reset_best_metric_on_resume", False)):
            best_val = float("inf")
            print("reset_best_metric_on_resume=true")
        start_epoch = resume_state.epoch + 1
        resumed_global_step = resume_state.global_step
        print(f"resumed_from={resolve_path(args.resume)} start_epoch={start_epoch}")

    trainer = SPADEGANTrainer(
        generator=generator,
        shadow_renderer=shadow_renderer,
        discriminator=discriminator,
        height_detail_discriminator=height_detail_discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        height_detail_discriminator_optimizer=height_detail_discriminator_optimizer,
        height_loss=height_loss,
        device=device,
        perceptual_loss=perceptual_loss,
        terrain_shape_loss=terrain_shape_loss,
        structure_feature_loss=structure_feature_loss,
        adversarial_weight=float(loss_config.get("adversarial", 1.0)),
        height_detail_adversarial_weight=float(loss_config.get("height_detail_adversarial", 0.0)),
        height_detail_feature_matching_weight=float(loss_config.get("height_detail_feature_matching", 0.0)),
        feature_matching_weight=float(loss_config.get("feature_matching", 10.0)),
        height_weight=float(loss_config.get("height", 2.0)),
        texture_weight=float(loss_config.get("texture", 1.0)),
        height_perceptual_weight=float(loss_config.get("height_perceptual", 0.0)),
        texture_perceptual_weight=float(loss_config.get("texture_perceptual", 0.0)),
        lab_gamut_weight=float(loss_config.get("lab_gamut", 0.0)),
        rgb_color_weight=float(loss_config.get("rgb_color", 0.0)),
        noise_regularization_weight=float(loss_config.get("noise_regularization", 0.0)),
        noise_regularization_start_block=int(loss_config.get("noise_regularization_start_block", 4)),
        disable_generator_above=float(train_config.get("disable_generator_above", 2.0)),
        enable_generator_below=float(train_config.get("enable_generator_below", 1.5)),
        max_discriminator_only_steps=_optional_int(train_config.get("max_discriminator_only_steps")),
        min_discriminator_steps_per_generator=int(train_config.get("min_discriminator_steps_per_generator", 0)),
        generator_warmup_epochs=int(train_config.get("generator_warmup_epochs", 0)),
        reconstruction_warmup_epochs=int(train_config.get("reconstruction_warmup_epochs", 0)),
        warmup_texture_weight=_optional_float(train_config.get("warmup_texture_weight")),
        structure_detail_warmup_epochs=int(train_config.get("structure_detail_warmup_epochs", 0)),
        structure_detail_ramp_epochs=int(train_config.get("structure_detail_ramp_epochs", 0)),
        dynamic_generator_gating=bool(train_config.get("dynamic_generator_gating", True)),
        reset_generator_gate_each_epoch=bool(train_config.get("reset_generator_gate_each_epoch", False)),
        ema_decay=float(train_config.get("ema_decay", 0.0)),
        use_ema_for_validation=bool(train_config.get("use_ema_for_validation", True)),
        discriminator_augmentation=train_config.get("discriminator_augmentation"),
        adaptive_discriminator_augmentation=train_config.get("adaptive_discriminator_augmentation"),
        amp=bool(train_config.get("amp", False)),
        max_train_steps=args.max_train_steps
        if args.max_train_steps is not None
        else _optional_int(train_config.get("max_train_steps")),
        max_val_steps=args.max_val_steps
        if args.max_val_steps is not None
        else _optional_int(train_config.get("max_val_steps")),
        clip_grad_norm=_optional_float(train_config.get("clip_grad_norm")),
        height_domain=str(train_config.get("height_domain", _height_domain_from_model(model_config))),
        channels_last=bool(train_config.get("channels_last", False)),
    )
    if resume_state is not None and "trainer" in resume_state.payload:
        trainer.load_state_dict(resume_state.payload["trainer"])
    else:
        trainer.global_step = resumed_global_step

    target_epochs = args.epochs if args.epochs is not None else int(train_config.get("epochs", 1))
    periodic_checkpoint_interval = max(0, int(train_config.get("periodic_checkpoint_interval", 0)))
    save_best_detail_checkpoint = bool(train_config.get("save_best_detail_checkpoint", True))
    last_completed_epoch = start_epoch - 1
    best_detail_ratio = -float("inf")
    if resume_state is not None:
        best_detail_ratio = float(resume_state.payload.get("best_detail_ratio", best_detail_ratio))
    try:
        if start_epoch >= target_epochs:
            print(f"training_already_complete start_epoch={start_epoch} target_epochs={target_epochs}")
            return
        for epoch in range(start_epoch, target_epochs):
            _apply_learning_rate_schedule(
                epoch,
                target_epochs,
                train_config,
                generator_optimizer,
                discriminator_optimizer,
                height_detail_discriminator_optimizer,
            )
            active_phase_name = _apply_training_phase(
                epoch,
                train_config,
                loss_config,
                trainer,
                generator_optimizer,
                discriminator_optimizer,
                height_detail_discriminator_optimizer,
            )
            train_metrics = trainer.train_one_epoch(epoch)
            val_metrics = trainer.validate(epoch)
            val_score = _validation_selection_score(val_metrics, train_config)
            print(
                f"epoch={epoch} "
                f"phase={active_phase_name} "
                f"g_loss={train_metrics.generator_loss:.6f} "
                f"d_loss={train_metrics.discriminator_loss:.6f} "
                f"h_d_loss={train_metrics.height_detail_discriminator_loss:.6f} "
                f"adv={train_metrics.adversarial_loss:.6f} "
                f"h_adv={train_metrics.height_detail_adversarial_loss:.6f} "
                f"h_fm={train_metrics.height_detail_feature_matching_loss:.6f} "
                f"fm={train_metrics.feature_matching_loss:.6f} "
                f"h={train_metrics.height_loss:.6f} "
                f"tex={train_metrics.texture_loss:.6f} "
                f"h_vgg={train_metrics.height_perceptual_loss:.6f} "
                f"tex_vgg={train_metrics.texture_perceptual_loss:.6f} "
                f"lab_gamut={train_metrics.lab_gamut_loss:.6f} "
                f"rgb_color={train_metrics.rgb_color_loss:.6f} "
                f"struct_scale={train_metrics.structure_detail_scale:.3f} "
                f"noise_reg={train_metrics.noise_regularization_loss:.6f} "
                f"shape={train_metrics.terrain_shape_loss:.6f} "
                f"branch={train_metrics.terrain_branch_morphology_loss:.6f} "
                f"struct_feat={train_metrics.terrain_structure_feature_loss:.6f} "
                f"val_h={val_metrics.val_height_l1:.6f} "
                f"val_tex={val_metrics.val_texture_l1:.6f} "
                f"val_grad={val_metrics.val_gradient_l1:.6f} "
                f"val_lap={val_metrics.val_laplacian_l1:.6f} "
                f"detail={val_metrics.val_generated_detail_energy:.6f}/{val_metrics.val_target_detail_energy:.6f} "
                f"water={val_metrics.val_water_flatness:.6f} "
                f"coast={val_metrics.val_coast_sharpness:.6f} "
                f"noise={val_metrics.val_noise_height_l1:.6f} "
                f"d_real={train_metrics.discriminator_real_confidence:.3f} "
                f"aug_p={train_metrics.augmentation_probability:.3f} "
                f"score={val_score:.6f} "
                f"g_updates={train_metrics.generator_updates} "
                f"d_updates={train_metrics.discriminator_updates}"
            )
            _append_metrics_row(
                metrics_path,
                epoch,
                trainer.global_step,
                train_metrics,
                val_metrics,
                val_score,
                phase_name=active_phase_name,
            )
            detail_ratio_improved = val_metrics.val_detail_energy_ratio > best_detail_ratio
            if detail_ratio_improved:
                best_detail_ratio = val_metrics.val_detail_energy_ratio
            state = _build_state(
                generator=generator,
                shadow_renderer=shadow_renderer,
                discriminator=discriminator,
                height_detail_discriminator=height_detail_discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                height_detail_discriminator_optimizer=height_detail_discriminator_optimizer,
                epoch=epoch,
                global_step=trainer.global_step,
                best_metric=min(best_val, val_score),
                best_detail_ratio=best_detail_ratio,
                trainer=trainer,
                config=raw,
            )
            checkpoint_manager.save_last(state)
            if periodic_checkpoint_interval > 0 and (
                epoch % periodic_checkpoint_interval == 0 or epoch == target_epochs - 1
            ):
                checkpoint_manager.save_epoch(state)
            if val_score < best_val:
                best_val = val_score
                checkpoint_manager.save_best(state)
            if save_best_detail_checkpoint and detail_ratio_improved:
                checkpoint_manager.save_best_detail(state)
            _save_preview_grid(
                generator=trainer.ema_generator
                if bool(train_config.get("use_ema_for_preview", train_config.get("use_ema_for_validation", True)))
                and trainer.ema_generator is not None
                else generator,
                shadow_renderer=trainer.ema_shadow_renderer
                if bool(train_config.get("use_ema_for_preview", train_config.get("use_ema_for_validation", True)))
                and trainer.ema_shadow_renderer is not None
                else shadow_renderer,
                loader=val_loader,
                device=device,
                output_path=resolve_path(preview_dir) / f"epoch_{epoch:04d}.png",
                seed=int(train_config.get("preview_seed", 12_345)),
                preview_config=preview_config,
                height_domain=str(train_config.get("height_domain", _height_domain_from_model(model_config))),
                amp=bool(train_config.get("amp", False)),
            )
            _save_diagnostic_previews(
                epoch=epoch,
                preview_config=preview_config,
                generator=trainer.ema_generator
                if bool(train_config.get("use_ema_for_preview", train_config.get("use_ema_for_validation", True)))
                and trainer.ema_generator is not None
                else generator,
                shadow_renderer=trainer.ema_shadow_renderer
                if bool(train_config.get("use_ema_for_preview", train_config.get("use_ema_for_validation", True)))
                and trainer.ema_shadow_renderer is not None
                else shadow_renderer,
                loader=val_loader,
                device=device,
                preview_dir=resolve_path(preview_dir),
                height_domain=str(train_config.get("height_domain", _height_domain_from_model(model_config))),
                amp=bool(train_config.get("amp", False)),
            )
            last_completed_epoch = epoch
    except KeyboardInterrupt:
        interrupted_epoch = last_completed_epoch if last_completed_epoch >= 0 else -1
        state = _build_state(
            generator=generator,
            shadow_renderer=shadow_renderer,
            discriminator=discriminator,
            height_detail_discriminator=height_detail_discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            height_detail_discriminator_optimizer=height_detail_discriminator_optimizer,
            epoch=interrupted_epoch,
            global_step=trainer.global_step,
            best_metric=best_val if best_val < float("inf") else None,
            best_detail_ratio=best_detail_ratio if best_detail_ratio > -float("inf") else None,
            trainer=trainer,
            config=raw,
        )
        checkpoint_manager.save_interrupted(state)
        raise


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_cuda_backend(train_config: dict, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if bool(train_config.get("cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True
    if bool(train_config.get("allow_tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _use_fused_adamw(train_config: dict, device: torch.device) -> bool:
    return device.type == "cuda" and bool(train_config.get("fused_adamw", False))


def _apply_learning_rate_schedule(
    epoch: int,
    target_epochs: int,
    train_config: dict,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    height_detail_discriminator_optimizer: torch.optim.Optimizer | None,
) -> None:
    schedule = train_config.get("lr_decay", {})
    if not schedule or not bool(schedule.get("enabled", False)):
        return
    start_epoch = int(schedule.get("start_epoch", max(0, int(target_epochs * 0.6))))
    if epoch < start_epoch:
        progress = 0.0
    else:
        decay_epochs = max(1, target_epochs - start_epoch)
        progress = min(1.0, (epoch - start_epoch) / decay_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    def apply(optimizer: torch.optim.Optimizer, base_key: str, min_key: str) -> None:
        base_lr = float(train_config.get(base_key, optimizer.param_groups[0]["lr"]))
        min_lr = float(schedule.get(min_key, base_lr))
        lr = min_lr + (base_lr - min_lr) * cosine
        for group in optimizer.param_groups:
            group["lr"] = lr

    apply(generator_optimizer, "generator_learning_rate", "generator_min_lr")
    apply(discriminator_optimizer, "discriminator_learning_rate", "discriminator_min_lr")
    if height_detail_discriminator_optimizer is not None:
        apply(
            height_detail_discriminator_optimizer,
            "height_detail_discriminator_learning_rate",
            "height_detail_discriminator_min_lr",
        )


def _apply_training_phase(
    epoch: int,
    train_config: dict,
    loss_config: dict,
    trainer: SPADEGANTrainer,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    height_detail_discriminator_optimizer: torch.optim.Optimizer | None,
) -> str:
    phase = _training_phase_for_epoch(train_config, epoch)
    if phase is None:
        return ""

    phase_training_config = {
        **dict(phase.get("training") or {}),
        **dict(phase.get("trainer") or {}),
    }
    merged_loss_config = {**loss_config, **dict(phase.get("loss") or {})}
    merged_training_config = {**train_config, **phase_training_config}
    trainer.apply_phase_overrides(
        loss_config=merged_loss_config,
        training_config=merged_training_config,
    )
    _apply_phase_learning_rates(
        phase,
        train_config,
        generator_optimizer,
        discriminator_optimizer,
        height_detail_discriminator_optimizer,
    )
    return str(phase.get("name", "unnamed"))


def _training_phase_for_epoch(train_config: dict, epoch: int) -> dict | None:
    phases = train_config.get("phases", [])
    if not phases:
        return None
    active_phase: dict | None = None
    for phase in phases:
        start_epoch = int(phase.get("start_epoch", 0))
        end_epoch = phase.get("end_epoch")
        if epoch < start_epoch:
            continue
        if end_epoch is not None and epoch >= int(end_epoch):
            continue
        active_phase = phase
    return active_phase


def _apply_phase_learning_rates(
    phase: dict,
    train_config: dict,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    height_detail_discriminator_optimizer: torch.optim.Optimizer | None,
) -> None:
    learning_rates = dict(phase.get("learning_rates", {}))

    def phase_lr(name: str, base_key: str) -> float | None:
        base_lr = float(train_config.get(base_key, 0.0))
        absolute_key = f"{name}_learning_rate"
        scale_key = f"{name}_lr_scale"
        if absolute_key in phase:
            return max(float(phase[absolute_key]), 0.0)
        if name in learning_rates:
            return max(float(learning_rates[name]), 0.0)
        if scale_key in phase:
            return max(base_lr * float(phase[scale_key]), 0.0)
        return None

    _set_optimizer_lr(
        generator_optimizer,
        phase_lr("generator", "generator_learning_rate"),
    )
    _set_optimizer_lr(
        discriminator_optimizer,
        phase_lr("discriminator", "discriminator_learning_rate"),
    )
    if height_detail_discriminator_optimizer is not None:
        _set_optimizer_lr(
            height_detail_discriminator_optimizer,
            phase_lr(
                "height_detail_discriminator",
                "height_detail_discriminator_learning_rate",
            ),
        )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float | None) -> None:
    if lr is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = lr


def _build_shadow_renderer(shadow_config: dict) -> torch.nn.Module:
    mode = str(shadow_config.get("mode", "hybrid_hillshade"))
    if mode in {"original_lab", "original_lab_rgb"}:
        return OriginalLabShadowRenderer(
            base_channels=int(shadow_config.get("base_channels", 32)),
            slope_scale=float(shadow_config.get("slope_scale", 1.0)),
            residual_strength=float(shadow_config.get("residual_strength", 1.0)),
            zero_init_last=bool(shadow_config.get("zero_init_last", False)),
            input_is_raw=bool(shadow_config.get("input_is_raw", False)),
        )
    if mode in {"original_luminance", "original_lab_like"}:
        return OriginalLuminanceShadowRenderer(
            base_channels=int(shadow_config.get("base_channels", 32)),
            slope_scale=float(shadow_config.get("slope_scale", 1.0)),
            residual_strength=float(shadow_config.get("residual_strength", 1.0)),
            zero_init_last=bool(shadow_config.get("zero_init_last", False)),
        )
    if mode == "original_learned":
        return OriginalLearnedShadowRenderer(
            base_channels=int(shadow_config.get("base_channels", 32)),
            slope_scale=float(shadow_config.get("slope_scale", 1.0)),
            residual_strength=float(shadow_config.get("residual_strength", 1.0)),
            zero_init_last=bool(shadow_config.get("zero_init_last", False)),
        )
    if mode != "hybrid_hillshade":
        raise ValueError(f"Unsupported shadow mode: {mode}")
    return LearnedShadowRenderer(
        base_channels=int(shadow_config.get("base_channels", 16)),
        slope_scale=float(shadow_config.get("slope_scale", 8.0)),
        azimuth_degrees=float(shadow_config.get("azimuth_degrees", 315.0)),
        altitude_degrees=float(shadow_config.get("altitude_degrees", 45.0)),
        hillshade_strength=float(shadow_config.get("hillshade_strength", 0.7)),
        residual_strength=float(shadow_config.get("residual_strength", 0.2)),
    )


def _build_height_detail_discriminator(discriminator_config: dict, model_config: dict) -> torch.nn.Module | None:
    detail_config = discriminator_config.get("height_detail", {})
    if not detail_config or not bool(detail_config.get("enabled", False)):
        return None
    return MultiScaleHeightDetailDiscriminator(
        semantic_channels=int(model_config.get("input_channels", 7)),
        base_channels=int(detail_config.get("base_channels", 24)),
        scales=tuple(float(scale) for scale in detail_config.get("scales", [1.0, 0.5, 0.25])),
        use_spectral_norm=bool(detail_config.get("use_spectral_norm", discriminator_config.get("use_spectral_norm", True))),
    )


def _height_domain_from_model(model_config: dict) -> str:
    return "zero_one" if str(model_config.get("height_activation", "sigmoid")) == "sigmoid" else "signed"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _bool_config(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _build_perceptual_loss(loss_config: dict) -> VGGPerceptualLoss | None:
    height_weight = float(loss_config.get("height_perceptual", 0.0))
    texture_weight = float(loss_config.get("texture_perceptual", 0.0))
    if height_weight <= 0.0 and texture_weight <= 0.0:
        return None
    layer_ids = tuple(int(layer_id) for layer_id in loss_config.get("vgg_layers", [3, 8, 17]))
    return VGGPerceptualLoss(
        layer_ids=layer_ids,
        pretrained=bool(loss_config.get("vgg_pretrained", True)),
    )


def _build_terrain_shape_loss(loss_config: dict) -> TerrainShapeLoss | None:
    criterion = TerrainShapeLoss(
        detail_energy_weight=float(loss_config.get("terrain_detail_energy", 0.0)),
        water_flatness_weight=float(loss_config.get("terrain_water_flatness", 0.0)),
        coast_sharpness_weight=float(loss_config.get("terrain_coast_sharpness", 0.0)),
        water_class_index=int(loss_config.get("water_class_index", 0)),
        detail_energy_mode=str(loss_config.get("terrain_detail_energy_mode", "global")),
        detail_energy_pooling=_detail_energy_pooling_config(loss_config.get("terrain_detail_energy_pooling", 16)),
        detail_energy_log_scale=float(loss_config.get("terrain_detail_energy_log_scale", 0.0)),
        detail_energy_min_land_fraction=float(loss_config.get("terrain_detail_energy_min_land_fraction", 0.05)),
        detail_energy_moment_weight=float(loss_config.get("terrain_detail_energy_moment_weight", 0.0)),
        detail_energy_class_weights=_optional_float_tuple(loss_config.get("terrain_detail_energy_class_weights")),
        branch_morphology_weight=float(loss_config.get("terrain_branch_morphology", 0.0)),
        branch_morphology_pooling=_detail_energy_pooling_config(
            loss_config.get(
                "terrain_branch_morphology_pooling",
                loss_config.get("terrain_detail_energy_pooling", 16),
            )
        ),
        branch_morphology_log_scale=float(loss_config.get("terrain_branch_morphology_log_scale", 24.0)),
        branch_morphology_coherence_weight=float(
            loss_config.get("terrain_branch_morphology_coherence_weight", 0.5)
        ),
    )
    return criterion if criterion.enabled else None


def _build_structure_feature_loss(loss_config: dict) -> StructureFeatureSupervisionLoss | None:
    configured = "terrain_structure_feature_supervision" in loss_config
    criterion = StructureFeatureSupervisionLoss(
        weight=float(loss_config.get("terrain_structure_feature_supervision", 0.0)),
        channel_weights=_optional_float_tuple(
            loss_config.get("terrain_structure_feature_channel_weights"),
            default=(1.0, 1.0, 0.7, 0.8),
        ),
        pooling=_detail_energy_pooling_config(loss_config.get("terrain_structure_feature_pooling", [4, 8, 16])),
        log_scale=float(loss_config.get("terrain_structure_feature_log_scale", 24.0)),
        min_land_fraction=float(loss_config.get("terrain_structure_feature_min_land_fraction", 0.05)),
        water_class_index=int(loss_config.get("water_class_index", 0)),
        semantic_class_channels=int(loss_config.get("terrain_structure_feature_semantic_class_channels", 7)),
        detail_budget_channel=int(loss_config.get("terrain_structure_feature_detail_budget_channel", 8)),
        detail_budget_floor=float(loss_config.get("terrain_structure_feature_detail_budget_floor", 0.25)),
    )
    return criterion if configured or criterion.enabled else None


def _detail_energy_pooling_config(value: object) -> int | tuple[int, ...]:
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    return int(value)


def _optional_float_tuple(value: object, default: tuple[float, ...] | None = None) -> tuple[float, ...] | None:
    if value is None:
        return default
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    return (float(value),)


def _validation_selection_score(val_metrics: SPADEGANEpochMetrics, train_config: dict) -> float:
    score = val_metrics.val_height_l1 + val_metrics.val_texture_l1
    target_detail_ratio = _optional_float(train_config.get("best_score_detail_target_ratio"))
    detail_under_weight = float(train_config.get("best_score_detail_under_weight", 0.0))
    if target_detail_ratio is not None and detail_under_weight > 0.0:
        detail_shortfall = max(0.0, target_detail_ratio - val_metrics.val_detail_energy_ratio)
        score += detail_under_weight * detail_shortfall

    max_detail_ratio = _optional_float(train_config.get("best_score_detail_max_ratio"))
    detail_over_weight = float(train_config.get("best_score_detail_over_weight", 0.0))
    if max_detail_ratio is not None and detail_over_weight > 0.0:
        detail_excess = max(0.0, val_metrics.val_detail_energy_ratio - max_detail_ratio)
        score += detail_over_weight * detail_excess
    texture_ceiling = _optional_float(train_config.get("best_score_texture_ceiling"))
    texture_over_weight = float(train_config.get("best_score_texture_over_weight", 0.0))
    if texture_ceiling is not None and texture_over_weight > 0.0:
        score += texture_over_weight * max(0.0, val_metrics.val_texture_l1 - texture_ceiling)
    height_ceiling = _optional_float(train_config.get("best_score_height_ceiling"))
    height_over_weight = float(train_config.get("best_score_height_over_weight", 0.0))
    if height_ceiling is not None and height_over_weight > 0.0:
        score += height_over_weight * max(0.0, val_metrics.val_height_l1 - height_ceiling)
    return score


def _build_state(
    generator: torch.nn.Module,
    shadow_renderer: torch.nn.Module,
    discriminator: torch.nn.Module,
    height_detail_discriminator: torch.nn.Module | None,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    height_detail_discriminator_optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    best_detail_ratio: float | None,
    trainer: SPADEGANTrainer | None,
    config: dict,
) -> CheckpointState:
    payload = {
        "model": generator.state_dict(),
        "shadow_renderer": shadow_renderer.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "rng": capture_rng_state(),
        "config": config,
        "best_detail_ratio": best_detail_ratio,
    }
    if height_detail_discriminator is not None:
        payload["height_detail_discriminator"] = height_detail_discriminator.state_dict()
    if height_detail_discriminator_optimizer is not None:
        payload["height_detail_discriminator_optimizer"] = height_detail_discriminator_optimizer.state_dict()
    if trainer is not None:
        trainer_state = trainer.state_dict()
        payload["trainer"] = trainer_state
        if "ema_generator" in trainer_state:
            payload["model_ema"] = trainer_state["ema_generator"]
        if "ema_shadow_renderer" in trainer_state:
            payload["shadow_renderer_ema"] = trainer_state["ema_shadow_renderer"]
    return CheckpointState(
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        payload=payload,
    )


def _restore_state(
    state: CheckpointState,
    generator: torch.nn.Module,
    shadow_renderer: torch.nn.Module,
    discriminator: torch.nn.Module,
    height_detail_discriminator: torch.nn.Module | None,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    height_detail_discriminator_optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> None:
    load_compatible_state_dict(generator, state.payload["model"])
    load_compatible_state_dict(shadow_renderer, state.payload["shadow_renderer"])
    load_compatible_state_dict(discriminator, state.payload["discriminator"])
    if height_detail_discriminator is not None and "height_detail_discriminator" in state.payload:
        load_compatible_state_dict(height_detail_discriminator, state.payload["height_detail_discriminator"])
    generator_optimizer.load_state_dict(state.payload["generator_optimizer"])
    discriminator_optimizer.load_state_dict(state.payload["discriminator_optimizer"])
    if (
        height_detail_discriminator_optimizer is not None
        and "height_detail_discriminator_optimizer" in state.payload
    ):
        height_detail_discriminator_optimizer.load_state_dict(state.payload["height_detail_discriminator_optimizer"])
    move_optimizer_state_to_device(generator_optimizer, device)
    move_optimizer_state_to_device(discriminator_optimizer, device)
    if height_detail_discriminator_optimizer is not None:
        move_optimizer_state_to_device(height_detail_discriminator_optimizer, device)
    if "rng" in state.payload:
        restore_rng_state(state.payload["rng"])


def _load_compatible_module_state(module: torch.nn.Module, source_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    return load_compatible_state_dict(module, source_state)


def _save_preview(
    generator: torch.nn.Module,
    shadow_renderer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    seed: int,
    height_domain: str,
    amp: bool,
    zero_noise: bool = False,
) -> None:
    was_generator_training = generator.training
    was_shadow_training = shadow_renderer.training
    generator.eval()
    shadow_renderer.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        biome = batch.biome_onehot.to(device)
        target_height = batch.height.to(device)
        if height_domain == "signed":
            target_height = target_height * 2.0 - 1.0
        target_texture = batch.texture.to(device)
        preview_generated, shadowed, _ = _generate_preview_outputs(
            generator=generator,
            shadow_renderer=shadow_renderer,
            biome=biome,
            device=device,
            seed=seed,
            zero_noise=zero_noise,
            amp=amp,
        )
    save_terrain_gan_preview(
        output_path,
        biome.cpu(),
        target_texture.cpu(),
        target_height.cpu(),
        preview_generated.cpu(),
        shadowed.cpu(),
    )
    if was_generator_training:
        generator.train()
    if was_shadow_training:
        shadow_renderer.train()


def _save_preview_grid(
    generator: torch.nn.Module,
    shadow_renderer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    seed: int,
    preview_config: dict,
    height_domain: str,
    amp: bool,
) -> None:
    was_generator_training = generator.training
    was_shadow_training = shadow_renderer.training
    generator.eval()
    shadow_renderer.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        sample_count = min(
            int(preview_config.get("summary_samples", 3)),
            int(batch.biome_onehot.shape[0]),
        )
        biome = batch.biome_onehot[:sample_count].to(device)
        target_height = batch.height[:sample_count].to(device)
        if height_domain == "signed":
            target_height = target_height * 2.0 - 1.0
        target_texture = batch.texture[:sample_count].to(device)

        variant_specs: list[tuple[int, bool]] = []
        if bool(preview_config.get("summary_include_zero_noise", True)):
            variant_specs.append((int(preview_config.get("zero_noise_seed", 0)), True))
        summary_seed_values = preview_config.get("summary_seed_values")
        if summary_seed_values is None:
            summary_seed_values = preview_config.get("multi_seed_values", [seed])
        max_seed_variants = max(1, int(preview_config.get("summary_max_seed_variants", 2)))
        for summary_seed in list(summary_seed_values)[:max_seed_variants]:
            variant_specs.append((int(summary_seed), False))

        variants: list[tuple[torch.Tensor, torch.Tensor]] = []
        for variant_seed, zero_noise in variant_specs:
            generated, shadowed, _ = _generate_preview_outputs(
                generator=generator,
                shadow_renderer=shadow_renderer,
                biome=biome,
                device=device,
                seed=variant_seed,
                zero_noise=zero_noise,
                amp=amp,
            )
            variants.append((generated.cpu(), shadowed.cpu()))

    save_terrain_gan_preview_grid(
        output_path,
        biome.cpu(),
        target_texture.cpu(),
        target_height.cpu(),
        variants,
    )
    if was_generator_training:
        generator.train()
    if was_shadow_training:
        shadow_renderer.train()


def _save_diagnostic_previews(
    epoch: int,
    preview_config: dict,
    generator: torch.nn.Module,
    shadow_renderer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    preview_dir: Path,
    height_domain: str,
    amp: bool,
) -> None:
    if not bool(preview_config.get("save_individual_previews", False)):
        return
    zero_interval = int(preview_config.get("zero_noise_interval", 0))
    if zero_interval > 0 and epoch % zero_interval == 0:
        _save_preview(
            generator=generator,
            shadow_renderer=shadow_renderer,
            loader=loader,
            device=device,
            output_path=preview_dir / f"epoch_{epoch:04d}_zero_noise.png",
            seed=int(preview_config.get("zero_noise_seed", 0)),
            height_domain=height_domain,
            amp=amp,
            zero_noise=True,
        )
    seed_interval = int(preview_config.get("multi_seed_interval", 0))
    if seed_interval <= 0 or epoch % seed_interval != 0:
        return
    for seed in preview_config.get("multi_seed_values", []):
        seed = int(seed)
        _save_preview(
            generator=generator,
            shadow_renderer=shadow_renderer,
            loader=loader,
            device=device,
            output_path=preview_dir / f"epoch_{epoch:04d}_seed_{seed}.png",
            seed=seed,
            height_domain=height_domain,
            amp=amp,
        )


def _append_metrics_row(
    metrics_path: Path,
    epoch: int,
    global_step: int,
    train_metrics: SPADEGANEpochMetrics,
    val_metrics: SPADEGANEpochMetrics,
    selection_score: float | None = None,
    phase_name: str | None = None,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _spade_metrics_fieldnames()
    _migrate_metrics_header_if_needed(metrics_path, fieldnames)
    write_header = not metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "phase": phase_name or "",
                "global_step": global_step,
                "train_generator_loss": train_metrics.generator_loss,
                "train_discriminator_loss": train_metrics.discriminator_loss,
                "train_height_detail_discriminator_loss": train_metrics.height_detail_discriminator_loss,
                "train_adversarial_loss": train_metrics.adversarial_loss,
                "train_height_detail_adversarial_loss": train_metrics.height_detail_adversarial_loss,
                "train_height_detail_feature_matching_loss": train_metrics.height_detail_feature_matching_loss,
                "train_feature_matching_loss": train_metrics.feature_matching_loss,
                "train_height_loss": train_metrics.height_loss,
                "train_texture_loss": train_metrics.texture_loss,
                "train_height_perceptual_loss": train_metrics.height_perceptual_loss,
                "train_texture_perceptual_loss": train_metrics.texture_perceptual_loss,
                "train_lab_gamut_loss": train_metrics.lab_gamut_loss,
                "train_rgb_color_loss": train_metrics.rgb_color_loss,
                "train_structure_detail_scale": train_metrics.structure_detail_scale,
                "train_noise_regularization_loss": train_metrics.noise_regularization_loss,
                "train_terrain_shape_loss": train_metrics.terrain_shape_loss,
                "train_terrain_detail_loss": train_metrics.terrain_detail_loss,
                "train_terrain_branch_morphology_loss": train_metrics.terrain_branch_morphology_loss,
                "train_terrain_structure_feature_loss": train_metrics.terrain_structure_feature_loss,
                "train_terrain_water_flatness_loss": train_metrics.terrain_water_flatness_loss,
                "train_terrain_coast_sharpness_loss": train_metrics.terrain_coast_sharpness_loss,
                "val_height_l1": val_metrics.val_height_l1,
                "val_texture_l1": val_metrics.val_texture_l1,
                "val_gradient_l1": val_metrics.val_gradient_l1,
                "val_laplacian_l1": val_metrics.val_laplacian_l1,
                "val_target_detail_energy": val_metrics.val_target_detail_energy,
                "val_generated_detail_energy": val_metrics.val_generated_detail_energy,
                "val_detail_energy_ratio": val_metrics.val_detail_energy_ratio,
                "val_water_flatness": val_metrics.val_water_flatness,
                "val_target_water_flatness": val_metrics.val_target_water_flatness,
                "val_coast_sharpness": val_metrics.val_coast_sharpness,
                "val_target_coast_sharpness": val_metrics.val_target_coast_sharpness,
                "val_noise_height_l1": val_metrics.val_noise_height_l1,
                "discriminator_real_confidence": train_metrics.discriminator_real_confidence,
                "augmentation_probability": train_metrics.augmentation_probability,
                "selection_score": selection_score
                if selection_score is not None
                else val_metrics.val_height_l1 + val_metrics.val_texture_l1,
                "generator_updates": train_metrics.generator_updates,
                "discriminator_updates": train_metrics.discriminator_updates,
            }
        )


def _spade_metrics_fieldnames() -> list[str]:
    return [
        "epoch",
        "phase",
        "global_step",
        "train_generator_loss",
        "train_discriminator_loss",
        "train_height_detail_discriminator_loss",
        "train_adversarial_loss",
        "train_height_detail_adversarial_loss",
        "train_height_detail_feature_matching_loss",
        "train_feature_matching_loss",
        "train_height_loss",
        "train_texture_loss",
        "train_height_perceptual_loss",
        "train_texture_perceptual_loss",
        "train_lab_gamut_loss",
        "train_rgb_color_loss",
        "train_structure_detail_scale",
        "train_noise_regularization_loss",
        "train_terrain_shape_loss",
        "train_terrain_detail_loss",
        "train_terrain_branch_morphology_loss",
        "train_terrain_structure_feature_loss",
        "train_terrain_water_flatness_loss",
        "train_terrain_coast_sharpness_loss",
        "val_height_l1",
        "val_texture_l1",
        "val_gradient_l1",
        "val_laplacian_l1",
        "val_target_detail_energy",
        "val_generated_detail_energy",
        "val_detail_energy_ratio",
        "val_water_flatness",
        "val_target_water_flatness",
        "val_coast_sharpness",
        "val_target_coast_sharpness",
        "val_noise_height_l1",
        "discriminator_real_confidence",
        "augmentation_probability",
        "selection_score",
        "generator_updates",
        "discriminator_updates",
    ]


def _migrate_metrics_header_if_needed(metrics_path: Path, fieldnames: list[str]) -> None:
    if not metrics_path.exists():
        return
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        existing_fieldnames = reader.fieldnames or []
        if existing_fieldnames == fieldnames:
            return
        rows = list(reader)
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _make_preview_noise_map(
    generator: torch.nn.Module,
    biome: torch.Tensor,
    preview_generator: torch.Generator,
) -> torch.Tensor | None:
    if uses_original_fresh_block_noise(generator):
        return None
    if getattr(generator, "noise_map_mode", "") == "terrain_fractal":
        return make_terrain_noise_map(
            biome.shape[0],
            biome.shape[-2:],
            biome.device,
            biome.dtype,
            randomize=True,
            generator=preview_generator,
        )
    return torch.randn(
        biome.shape[0],
        1,
        biome.shape[-2],
        biome.shape[-1],
        device=biome.device,
        dtype=biome.dtype,
        generator=preview_generator,
    )


def _generate_preview_outputs(
    generator: torch.nn.Module,
    shadow_renderer: torch.nn.Module,
    biome: torch.Tensor,
    device: torch.device,
    seed: int,
    zero_noise: bool,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preview_generator = torch.Generator(device=device)
    preview_generator.manual_seed(seed)
    if zero_noise:
        noise = torch.zeros(
            biome.shape[0],
            getattr(generator, "noise_dim", 16),
            device=device,
            dtype=biome.dtype,
        )
        noise_map: torch.Tensor | None = (
            None
            if uses_internal_original_noise(generator)
            else torch.zeros(
                biome.shape[0],
                1,
                biome.shape[-2],
                biome.shape[-1],
                device=device,
                dtype=biome.dtype,
            )
        )
    else:
        noise = sample_generator_latent(
            generator,
            biome.shape[0],
            device,
            biome.dtype,
            rng=preview_generator,
        )
        noise_map = _make_preview_noise_map(generator, biome, preview_generator)
    base_generated: torch.Tensor | None = None
    shadowed: torch.Tensor | None = None
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        if not zero_noise and uses_original_fresh_block_noise(generator, noise_map):
            generated_output = _forward_with_seeded_internal_noise(
                generator,
                biome,
                noise,
                seed,
            )
            if isinstance(generated_output, tuple):
                generated, base_generated = generated_output
            else:
                generated = generated_output
        else:
            if getattr(generator, "height_detail_shadow_source", "final") == "base":
                generated, shadowed = render_generator_terrain(generator, shadow_renderer, biome, noise, noise_map)
            else:
                generated = generator(biome, noise=noise, noise_map=noise_map)
                shadowed = shadow_renderer(generated)
        if base_generated is not None:
            shadowed = render_shadowed_terrain(shadow_renderer, generated, base_generated)
        elif shadowed is None:
            shadowed = shadow_renderer(generated)
        preview_generated = generated
        texture_to_rgb = getattr(shadow_renderer, "texture_to_rgb", None)
        if callable(texture_to_rgb):
            preview_generated = generated.clone()
            preview_generated[:, :3] = texture_to_rgb(generated[:, :3])
            if getattr(generator, "output_activation_mode", "") == "original_raw":
                preview_generated[:, 3:4] = shadowed[:, 3:4]
    return preview_generated, shadowed, noise


def _forward_with_seeded_internal_noise(
    generator: torch.nn.Module,
    biome: torch.Tensor,
    noise: torch.Tensor,
    seed: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    was_training = generator.training
    cuda_devices: list[int] = []
    if biome.device.type == "cuda":
        cuda_devices = [biome.device.index if biome.device.index is not None else torch.cuda.current_device()]
    try:
        generator.train()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if biome.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            if getattr(generator, "height_detail_shadow_source", "final") == "base":
                forward_with_base = getattr(generator, "forward_with_base_output", None)
                if callable(forward_with_base):
                    return forward_with_base(biome, noise=noise, noise_map=None)
            return generator(biome, noise=noise, noise_map=None)
    finally:
        if not was_training:
            generator.eval()


if __name__ == "__main__":
    main()
