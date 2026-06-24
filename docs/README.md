# Documentation

This folder is intentionally compact for the public repository. The main
workflow is documented in the root `README.md`; this folder is reserved for
short project notes that are useful to users of the repository.

## Model Interface

- Input condition: `biome_onehot [1, 11, 256, 256]`.
- Latent input: `noise [1, 16]`.
- Output: `terrain [1, 4, 256, 256]`.
- Output channels `0..2`: generated RGB texture.
- Output channel `3`: generated height map.

The Unity runtime treats `train_noise_00..train_noise_06`, when present in an
exported model, as auxiliary replay inputs. They are not user-facing controls.
