# Configurations

Only the public entrypoint configs are kept in this folder:

| File | Purpose |
| --- | --- |
| `train_terrain_gan.yaml` | main training config for the feature-supervised terrain GAN |
| `export_onnx.yaml` | ONNX export and validation config |
| `biomes.yaml` | macro-biome class names |
| `palette.yaml` | default palette mapping |

Generated checkpoints, previews, reports, and exported ONNX files are written to
`artifacts/` unless another path is passed on the command line.
