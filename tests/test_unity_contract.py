"""Project-local Unity pipeline contract checks."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNITY_SCRIPTS = PROJECT_ROOT / "unity" / "Assets" / "Scripts"


def test_unity_contract_uses_project_local_scripts() -> None:
    assert UNITY_SCRIPTS.exists()
    project_settings = PROJECT_ROOT / "unity" / "ProjectSettings" / "ProjectSettings.asset"
    assert "productName: UnityTerrainGANerator" in project_settings.read_text(encoding="utf-8")


def test_unity_v12_4_inference_uses_latent_noise_not_spatial_noise_map() -> None:
    source = _read_unity_script("Inference/ONNXInferenceController.cs")

    assert "noiseMapInputName" not in source
    assert "CreateNoiseMapTensor" not in source
    assert "noise_map" not in source
    assert 'public string noiseInputName = "noise";' in source
    assert "public int noiseDim = 16;" in source
    assert "CreateNoiseTensor" in source


def test_unity_v12_4_profile_uses_11_input_channels() -> None:
    source = _read_unity_script("Inference/ONNXInferenceController.cs")

    assert "public int inputChannelCount = 11;" in source
    assert "ConfigureForV12StructureModel" in source
    assert "inputChannelCount = 11;" in source
    assert "heightChannelIndex = 3;" in source


def test_palette_encoder_supports_general_relief_v2_without_default_switch() -> None:
    source = _read_unity_script("Biomes/PaletteEncoder.cs")

    assert "GeneralReliefV2" in source
    assert "AppendV124ReliefControlMaps" in source
    assert "heightPriorByClass" in source
    assert "detailBudgetByClass" in source
    assert "BuildBoundaryDistance" in source
    assert "BuildSemanticComplexity" in source
    assert "CopyChannel(heightPrior, data, macroChannelCount, width, height);" in source
    assert "CopyChannel(detailBudget, data, macroChannelCount + 1, width, height);" in source
    assert "CopyChannel(boundary, data, macroChannelCount + 2, width, height);" in source
    assert "CopyChannel(complexity, data, macroChannelCount + 3, width, height);" in source


def test_terrain_generator_reads_height_from_terrain4_channel_3() -> None:
    source = _read_unity_script("Terrain/TerrainGenerator.cs")

    assert "public int generatedHeightChannel = 3;" in source
    assert "channelCount > 1 ? Mathf.Clamp(generatedHeightChannel, 0, channelCount - 1) : 0" in source
    assert "heightPostprocessor.TensorToHeights(outputData, width, height, heightChannel, channelCount)" in source


def test_unity_input_orientation_contract() -> None:
    source = _read_unity_script("Biomes/PaletteEncoder.cs")

    assert "public bool flipY = true;" in source
    assert "new TensorShape(1, palette.macroClassCount, map.Height, map.Width)" in source
    assert "((channel * height + y) * width) + x" in source

    height = 3
    width = 5
    macro_class = 2
    y = 1
    x = 4

    assert _nchw_index(macro_class, y, x, height, width) == 39


def _nchw_index(channel: int, y: int, x: int, height: int, width: int) -> int:
    return ((channel * height + y) * width) + x


def _read_unity_script(relative_path: str) -> str:
    return (UNITY_SCRIPTS / relative_path).read_text(encoding="utf-8")
