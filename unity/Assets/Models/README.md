# ONNX Models

This folder contains generator models imported by the Unity project.

| File | Notes |
| --- | --- |
| `spade_terrain_epoch116.onnx` | recommended balanced runtime model |
| `spade_terrain_epoch128.onnx` | detail-focused candidate |
| `spade_terrain_epoch90.onnx` | earlier candidate kept for comparison |

The epoch models expose `biome_onehot`, `noise`, and auxiliary
`train_noise_00..train_noise_06` inputs. The auxiliary inputs are handled by the
Unity profile and are not intended as user controls.
