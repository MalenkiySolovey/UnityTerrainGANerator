using Unity.InferenceEngine;
using UnityEngine;
using UnityTerrainGANerator.Biomes;
using UnityTerrainGANerator.Inference;
using UnityTerrainGANerator.TerrainGeneration;

namespace UnityTerrainGANerator.Preview
{
    [CreateAssetMenu(menuName = "UnityTerrainGANerator/Preview Profile")]
    public sealed class TerrainGANPreviewProfile : ScriptableObject
    {
        [Header("Inputs")]
        public BiomePalette palette;
        public Texture2D biomeTexture;
        public ModelAsset modelAsset;

        [Header("Inference")]
        public BackendType backendType = BackendType.CPU;
        public string biomeInputName = "biome_onehot";
        public string noiseInputName = "noise";
        public string terrainOutputName = "terrain";
        public bool useNoiseInputs = false;
        public int noiseSeed = 12345;
        [Range(0f, 1f)] public float noiseStrength = 1f;
        public LatentNoiseDistribution noiseDistribution = LatentNoiseDistribution.Uniform01;
        public int noiseDim = 16;
        public bool useTrainingNoiseInputs = true;
        public string trainingNoiseInputPrefix = "train_noise_";
        [Range(0f, 1f)] public float trainingNoiseStrength = 1f;
        public int inputResolution = 256;
        public int macroClassCount = 7;
        public int modelInputChannels = 11;

        [Header("Generated Output Channels")]
        public TerrainTextureMode textureMode = TerrainTextureMode.GeneratedTexture;
        public int generatedTextureRedChannel = 0;
        public int generatedTextureGreenChannel = 1;
        public int generatedTextureBlueChannel = 2;
        public int generatedHeightChannel = 3;
        [Range(0f, 2f)] public float textureSaturation = 1f;
        [Range(0f, 2f)] public float textureBrightness = 1f;
        [Range(0f, 2f)] public float textureContrast = 1f;
        [Range(0.25f, 3f)] public float textureGamma = 1f;
        public Color textureTint = Color.white;

        [Header("Orientation")]
        public bool invertOutputWidth = false;
        public bool invertOutputLength = false;

        [Header("Terrain")]
        public Vector3 terrainSize = new Vector3(1000f, 220f, 1000f);
        public int heightmapResolution = 257;
        public int alphamapResolution = 256;
        public bool useDelayedLod = false;
        public bool paintTerrainLayers = false;
        public bool useTiledGeneration = false;
        [Range(1, 8)] public int tileGrid = 1;
        [Range(0, 96)] public int tileOverlapPixels = 32;
        public bool usePerTileVariation = true;

        [Header("Object Placement")]
        public bool enableObjectPlacement = true;
        public bool clearObjectsWhenDisabled = true;
        public bool useLegacyShortcutPlacement = false;
        public bool useObjectRulesPlacement = true;
        [Range(8, 512)] public int objectPlacementResolution = 128;
        [Range(8, 512)] public int objectRulePlacementResolution = 128;
        [Min(0f)] public float objectRuleDensityScale = 1f;
        [Min(0)] public int maxObjectRuleInstances = 1200;

        [Header("Postprocess")]
        public bool clamp01 = true;
        public bool resize256To257 = true;
        public bool remapWithPercentiles = true;
        [Range(0f, 0.25f)] public float lowerPercentile = 0.02f;
        [Range(0.75f, 1f)] public float upperPercentile = 0.98f;
        [Range(0f, 1f)] public float outputMin = 0.02f;
        [Range(0f, 1f)] public float outputMax = 0.42f;
        [Min(0)] public int smoothingIterations = 0;
        [Range(0f, 1f)] public float smoothingStrength = 0.65f;

        public void ApplyTo(TerrainGenerator terrainGenerator)
        {
            if (terrainGenerator == null)
            {
                throw new System.ArgumentNullException(nameof(terrainGenerator));
            }

            if (terrainGenerator.paletteEncoder != null)
            {
                terrainGenerator.paletteEncoder.palette = palette;
            }

            if (terrainGenerator.inferenceController != null)
            {
                terrainGenerator.inferenceController.modelAsset = modelAsset;
                terrainGenerator.inferenceController.backendType = backendType;
                terrainGenerator.inferenceController.biomeInputName = biomeInputName;
                terrainGenerator.inferenceController.noiseInputName = noiseInputName;
                terrainGenerator.inferenceController.terrainOutputName = terrainOutputName;
                terrainGenerator.inferenceController.useNoiseInputs = useNoiseInputs;
                terrainGenerator.inferenceController.noiseSeed = noiseSeed;
                terrainGenerator.inferenceController.noiseStrength = noiseStrength;
                terrainGenerator.inferenceController.noiseDistribution = noiseDistribution;
                terrainGenerator.inferenceController.noiseDim = noiseDim;
                terrainGenerator.inferenceController.useTrainingNoiseInputs = useTrainingNoiseInputs;
                terrainGenerator.inferenceController.trainingNoiseInputPrefix = trainingNoiseInputPrefix;
                terrainGenerator.inferenceController.trainingNoiseStrength = trainingNoiseStrength;
                terrainGenerator.inferenceController.inputResolution = inputResolution;
                terrainGenerator.inferenceController.macroClassCount = macroClassCount;
                terrainGenerator.inferenceController.modelInputChannels = modelInputChannels;
                terrainGenerator.inferenceController.generatedHeightChannel = generatedHeightChannel;
            }

            if (terrainGenerator.heightPostprocessor != null)
            {
                HeightPostprocessor postprocessor = terrainGenerator.heightPostprocessor;
                postprocessor.clamp01 = clamp01;
                postprocessor.resize256To257 = resize256To257;
                postprocessor.targetHeightmapResolution = heightmapResolution;
                postprocessor.ConfigureOutputMirror(invertOutputWidth, invertOutputLength);
                postprocessor.remapWithPercentiles = remapWithPercentiles;
                postprocessor.lowerPercentile = lowerPercentile;
                postprocessor.upperPercentile = upperPercentile;
                postprocessor.outputMin = outputMin;
                postprocessor.outputMax = outputMax;
                postprocessor.smoothingIterations = smoothingIterations;
                postprocessor.smoothingStrength = smoothingStrength;
            }

            if (terrainGenerator.texturePostprocessor == null)
            {
                terrainGenerator.texturePostprocessor =
                    terrainGenerator.GetComponent<TerrainTexturePostprocessor>() ??
                    terrainGenerator.gameObject.AddComponent<TerrainTexturePostprocessor>();
            }

            if (terrainGenerator.texturePostprocessor != null)
            {
                terrainGenerator.texturePostprocessor.ConfigureOutputMirror(invertOutputWidth, invertOutputLength);
                terrainGenerator.texturePostprocessor.saturation = textureSaturation;
                terrainGenerator.texturePostprocessor.brightness = textureBrightness;
                terrainGenerator.texturePostprocessor.contrast = textureContrast;
                terrainGenerator.texturePostprocessor.gamma = textureGamma;
                terrainGenerator.texturePostprocessor.tint = textureTint;
            }

            if (terrainGenerator.terrainBuilder != null)
            {
                terrainGenerator.terrainBuilder.terrainSize = terrainSize;
                terrainGenerator.terrainBuilder.heightmapResolution = heightmapResolution;
                terrainGenerator.terrainBuilder.alphamapResolution = alphamapResolution;
                terrainGenerator.terrainBuilder.useDelayedLod = useDelayedLod;
            }

            terrainGenerator.useTiledGeneration = useTiledGeneration;
            terrainGenerator.tileGrid = tileGrid;
            terrainGenerator.tileOverlapPixels = tileOverlapPixels;
            terrainGenerator.usePerTileVariation = usePerTileVariation;
            terrainGenerator.objectPlacementEnabled = enableObjectPlacement;
            terrainGenerator.ConfigureOutputMirror(invertOutputWidth, invertOutputLength);
            terrainGenerator.generatedTextureRedChannel = generatedTextureRedChannel;
            terrainGenerator.generatedTextureGreenChannel = generatedTextureGreenChannel;
            terrainGenerator.generatedTextureBlueChannel = generatedTextureBlueChannel;
            terrainGenerator.generatedHeightChannel = generatedHeightChannel;
            terrainGenerator.textureMode = textureMode;
            terrainGenerator.paintTerrainLayers = textureMode == TerrainTextureMode.PaintedBiomeMap || paintTerrainLayers;

            if (enableObjectPlacement && terrainGenerator.vegetationSpawner == null)
            {
                terrainGenerator.vegetationSpawner =
                    terrainGenerator.GetComponent<VegetationSpawner>() ??
                    terrainGenerator.gameObject.AddComponent<VegetationSpawner>();
            }

            if (terrainGenerator.vegetationSpawner != null)
            {
                terrainGenerator.vegetationSpawner.palette = palette;
                terrainGenerator.vegetationSpawner.targetTerrain = terrainGenerator.terrainBuilder != null
                    ? terrainGenerator.terrainBuilder.targetTerrain
                    : terrainGenerator.vegetationSpawner.targetTerrain;
                terrainGenerator.vegetationSpawner.rebuildTerrainTrees = enableObjectPlacement && useLegacyShortcutPlacement;
                terrainGenerator.vegetationSpawner.spawnDetailPrefabs = enableObjectPlacement && useLegacyShortcutPlacement;
                terrainGenerator.vegetationSpawner.spawnRockPrefabs = enableObjectPlacement && useLegacyShortcutPlacement;
                terrainGenerator.vegetationSpawner.spawnPlacementRules = enableObjectPlacement && useObjectRulesPlacement;
                terrainGenerator.vegetationSpawner.placementResolution = objectPlacementResolution;
                terrainGenerator.vegetationSpawner.rulePlacementResolution = objectRulePlacementResolution;
                terrainGenerator.vegetationSpawner.ruleDensityScale = objectRuleDensityScale;
                terrainGenerator.vegetationSpawner.maxRuleObjects = maxObjectRuleInstances;
                if (!enableObjectPlacement && clearObjectsWhenDisabled)
                {
                    terrainGenerator.vegetationSpawner.ClearSpawnedObjects();
                }
            }

            terrainGenerator.ReapplyCachedTerrain();
        }
    }
}
