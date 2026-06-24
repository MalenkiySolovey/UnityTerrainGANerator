# UnityTerrainGANerator

> Archived diploma project. The repository is published as-is and is not
> planned for further maintenance.

UnityTerrainGANerator generates Unity Terrain data from a semantic biome map.
The project contains a Python training/export pipeline, exported ONNX generator
models, Unity runtime/editor tools, and contract tests for the model interface.

## Runtime Contract

The Unity runtime uses a fixed terrain-generation contract:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `biome_onehot` | `[1, 11, 256, 256]` | biome condition and relief-control channels |
| `noise` | `[1, 16]` | latent variation vector |
| `terrain` | `[1, 4, 256, 256]` | generated RGB texture and height |

The first seven input channels are macro-biome one-hot channels:

| Index | Macro biome |
| --- | --- |
| `0` | water |
| `1` | sand |
| `2` | grass |
| `3` | dirt |
| `4` | snow |
| `5` | rock |
| `6` | mountain |

The remaining channels describe broad relief controls:

| Index | Channel |
| --- | --- |
| `7` | `height_prior` |
| `8` | `detail_budget` |
| `9` | `boundary_distance` |
| `10` | `semantic_complexity` |

Output channels `0..2` are the generated RGB texture, and output channel `3`
is the generated height map.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `configs/` | public training and export YAML configs |
| `src/terrain_gan/` | Python package for data handling, model code, export, validation, and training utilities |
| `scripts/` | command-line entry points for training, ONNX export, and ONNX validation |
| `tests/` | Python contract tests for the model and Unity integration |
| `unity/` | Unity project with editor/runtime scripts, demo assets, and ONNX models |
| `data/` | local dataset mount point; raw data is not committed |
| `artifacts/` | generated checkpoints, previews, reports, and exports; runtime output is ignored by Git |
| `docs/` | short project notes for public use |

## Python Setup

Use Python 3.11 or 3.12. The pinned CUDA 12.1 PyTorch build in
`requirements.txt` is not available for newer Python versions.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Training

Place the training dataset under `data/raw/archive/_dataset`, prepare split CSV
files under `data/splits`, and run:

```powershell
python -m scripts.train_spade_terrain_gan --config configs\train_terrain_gan.yaml
```

Training outputs are written to `artifacts/checkpoints`, `artifacts/previews`,
and `artifacts/reports`.

## ONNX Export

Export a checkpoint with:

```powershell
python -m scripts.export_onnx `
  --config configs\export_onnx.yaml `
  --checkpoint artifacts\checkpoints\terrain_gan\best.pt `
  --output artifacts\onnx\terrain_gan.onnx
```

Validate an exported model:

```powershell
python -m scripts.validate_onnx `
  --config configs\export_onnx.yaml `
  --onnx unity\Assets\Models\spade_terrain_rgb_height_v12_4_feature_supervised_256.onnx `
  --mode terrain4
```

## Unity

Open the Unity project from the `unity/` folder. The main runtime/editor flow is:

1. prepare or import a biome map;
2. encode it with `PaletteEncoder`;
3. run an ONNX model through `ONNXInferenceController`;
4. post-process generated height and RGB channels;
5. apply the result to `TerrainData`.

The included ONNX files are stored in `unity/Assets/Models` so the Unity project
can be opened without downloading model assets separately.

## Tests

```powershell
python -m pytest
```

The tests check the Python ONNX contract and the Unity script contract.
