using Unity.InferenceEngine;
using UnityEngine;
using UnityTerrainGANerator.Biomes;
using UnityTerrainGANerator.Inference;

namespace UnityTerrainGANerator.TerrainGeneration
{
    public enum TerrainTextureMode
    {
        None,
        PaintedBiomeMap,
        GeneratedTexture
    }

    public sealed class TerrainGenerator : MonoBehaviour
    {
        public PaletteEncoder paletteEncoder;
        public ONNXInferenceController inferenceController;
        public HeightPostprocessor heightPostprocessor;
        public TerrainTexturePostprocessor texturePostprocessor;
        public TerrainBuilder terrainBuilder;
        public VegetationSpawner vegetationSpawner;
        public TerrainTextureMode textureMode = TerrainTextureMode.GeneratedTexture;
        public bool paintTerrainLayers = true;
        public int generatedTextureRedChannel = 0;
        public int generatedTextureGreenChannel = 1;
        public int generatedTextureBlueChannel = 2;
        public int generatedHeightChannel = 3;
        public bool useTiledGeneration = false;
        [Range(1, 8)] public int tileGrid = 1;
        [Range(0, 96)] public int tileOverlapPixels = 32;
        public bool usePerTileVariation = true;
        public bool objectPlacementEnabled = true;

        private Texture2D lastBiomeTexture;
        private Texture2D lastAppliedTexture;
        private Texture2D lastGeneratedTexture;
        private MicroBiomeMap lastTerrainBiomeMap;
        private float[] lastOutputData;
        private int lastChannelCount;
        private int lastWidth;
        private int lastHeight;
        private bool mirrorOutputWidth;
        private bool mirrorOutputLength;

        public bool HasGeneratedTerrain => lastOutputData != null;
        public bool HasTerrainBiomeMap => lastTerrainBiomeMap != null;
        public Texture2D LastBiomeTexture => lastBiomeTexture;
        public Texture2D LastAppliedTexture => lastAppliedTexture;
        public Texture2D LastGeneratedTexture => lastGeneratedTexture;

        public void GenerateFromBiomeTexture(Texture2D biomeTexture)
        {
            if (paletteEncoder == null || inferenceController == null || heightPostprocessor == null || terrainBuilder == null)
            {
                throw new System.InvalidOperationException("Terrain generation components are not fully assigned.");
            }

            if (textureMode == TerrainTextureMode.GeneratedTexture && texturePostprocessor == null)
            {
                texturePostprocessor = GetComponent<TerrainTexturePostprocessor>() ?? gameObject.AddComponent<TerrainTexturePostprocessor>();
            }

            if (useTiledGeneration && tileGrid > 1)
            {
                GenerateTiledFromBiomeTexture(biomeTexture);
                return;
            }

            int inputResolution = Mathf.Max(1, inferenceController.inputResolution);
            MicroBiomeMap terrainBiomeMap = BuildTerrainBiomeMap(biomeTexture, ResolveTerrainBiomeMapResolution(inputResolution, biomeTexture));
            using Tensor<float> biomeTensor = BuildModelInputTensor(biomeTexture, inputResolution);
            using Tensor<float> terrainTensor = inferenceController.RunTerrain(biomeTensor, null);

            lastBiomeTexture = biomeTexture;
            lastTerrainBiomeMap = terrainBiomeMap;
            CacheTerrainTensor(terrainTensor);
            ApplyCachedHeight();
            ApplyCachedTexture();

            if (vegetationSpawner != null)
            {
                if (vegetationSpawner.palette == null)
                {
                    vegetationSpawner.palette = paletteEncoder.palette;
                }

                if (vegetationSpawner.targetTerrain == null)
                {
                    vegetationSpawner.targetTerrain = terrainBuilder.targetTerrain;
                }

                if (objectPlacementEnabled)
                {
                    vegetationSpawner.Spawn(terrainBiomeMap);
                }
                else
                {
                    vegetationSpawner.ClearSpawnedObjects();
                }
            }
        }

        private void GenerateTiledFromBiomeTexture(Texture2D biomeTexture)
        {
            int inputResolution = Mathf.Max(1, inferenceController.inputResolution);
            int grid = Mathf.Clamp(tileGrid, 1, 8);
            int overlap = Mathf.Clamp(tileOverlapPixels, 0, Mathf.Max(0, inputResolution / 2 - 1));
            int stride = Mathf.Max(1, inputResolution - overlap);
            int assembledResolution = stride * (grid - 1) + inputResolution;
            int modelChannels = Mathf.Max(1, inferenceController.modelInputChannels);
            Debug.Log(
                $"Terrain GAN tiled generation: {grid}x{grid} ONNX tiles, input {inputResolution}px, " +
                $"overlap {overlap}px, assembled output {assembledResolution}x{assembledResolution}."
            );

            MicroBiomeMap globalModelInputMap = paletteEncoder.BuildMicroBiomeMap(
                biomeTexture,
                assembledResolution,
                assembledResolution);
            MicroBiomeMap terrainBiomeMap = BuildTerrainBiomeMap(
                biomeTexture,
                ResolveTerrainBiomeMapResolution(assembledResolution, biomeTexture));

            float[] assembledData = null;
            float[] weights = new float[assembledResolution * assembledResolution];
            int outputChannelCount = 0;
            int originalSeed = inferenceController.noiseSeed;

            try
            {
                for (int tileY = 0; tileY < grid; tileY++)
                {
                    for (int tileX = 0; tileX < grid; tileX++)
                    {
                        int startX = tileX * stride;
                        int startY = tileY * stride;
                        MicroBiomeMap tileMap = ExtractTile(globalModelInputMap, startX, startY, inputResolution);
                        using Tensor<float> biomeTensor = paletteEncoder.BuildModelInputTensor(tileMap, modelChannels);

                        if (usePerTileVariation && inferenceController.useNoiseInputs)
                        {
                            inferenceController.noiseSeed = BuildTileSeed(originalSeed, tileX, tileY, grid);
                        }

                        using Tensor<float> terrainTensor = inferenceController.RunTerrain(biomeTensor, null);
                        float[] tileData = terrainTensor.DownloadToArray();
                        int channelCount = terrainTensor.shape[1];
                        int tileHeight = terrainTensor.shape[2];
                        int tileWidth = terrainTensor.shape[3];
                        if (tileWidth != inputResolution || tileHeight != inputResolution)
                        {
                            throw new System.InvalidOperationException(
                                $"Tiled generation expects {inputResolution}x{inputResolution} model output, got {tileWidth}x{tileHeight}."
                            );
                        }

                        if (assembledData == null)
                        {
                            outputChannelCount = channelCount;
                            assembledData = new float[outputChannelCount * assembledResolution * assembledResolution];
                        }
                        else if (channelCount != outputChannelCount)
                        {
                            throw new System.InvalidOperationException("Tiled generation received inconsistent output channel counts.");
                        }

                        AccumulateTile(
                            assembledData,
                            weights,
                            tileData,
                            outputChannelCount,
                            inputResolution,
                            assembledResolution,
                            startX,
                            startY,
                            overlap,
                            tileX,
                            tileY,
                            grid);
                    }
                }
            }
            finally
            {
                inferenceController.noiseSeed = originalSeed;
            }

            if (assembledData == null)
            {
                throw new System.InvalidOperationException("Tiled generation did not produce any terrain output.");
            }

            NormalizeAssembledTiles(assembledData, weights, outputChannelCount, assembledResolution);

            lastBiomeTexture = biomeTexture;
            lastTerrainBiomeMap = terrainBiomeMap;
            lastOutputData = assembledData;
            lastChannelCount = outputChannelCount;
            lastWidth = assembledResolution;
            lastHeight = assembledResolution;
            ApplyCachedHeight();
            ApplyCachedTexture();

            if (vegetationSpawner != null)
            {
                if (vegetationSpawner.palette == null)
                {
                    vegetationSpawner.palette = paletteEncoder.palette;
                }

                if (vegetationSpawner.targetTerrain == null)
                {
                    vegetationSpawner.targetTerrain = terrainBuilder.targetTerrain;
                }

                if (objectPlacementEnabled)
                {
                    vegetationSpawner.Spawn(terrainBiomeMap);
                }
                else
                {
                    vegetationSpawner.ClearSpawnedObjects();
                }
            }
        }

        public void SetTextureMode(TerrainTextureMode mode)
        {
            textureMode = mode;
            paintTerrainLayers = mode == TerrainTextureMode.PaintedBiomeMap;
            if (HasGeneratedTerrain)
            {
                ReapplyCachedTerrain();
            }
        }

        public void ConfigureOutputMirror(bool mirrorWidth, bool mirrorLength)
        {
            mirrorOutputWidth = mirrorWidth;
            mirrorOutputLength = mirrorLength;
            if (heightPostprocessor != null)
            {
                heightPostprocessor.ConfigureOutputMirror(mirrorOutputWidth, mirrorOutputLength);
            }

            if (texturePostprocessor != null)
            {
                texturePostprocessor.ConfigureOutputMirror(mirrorOutputWidth, mirrorOutputLength);
            }
        }

        public void ReapplyCachedTerrain()
        {
            if (!HasGeneratedTerrain)
            {
                return;
            }

            ApplyCachedHeight();
            ApplyCachedTexture();
        }

        public void RespawnVegetation()
        {
            if (vegetationSpawner == null)
            {
                Debug.LogWarning("TerrainGenerator has no VegetationSpawner assigned.");
                return;
            }

            if (!objectPlacementEnabled)
            {
                vegetationSpawner.ClearSpawnedObjects();
                return;
            }

            if (lastTerrainBiomeMap == null)
            {
                Debug.LogWarning("No terrain biome map is cached yet. Generate terrain once before respawning objects.");
                return;
            }

            if (vegetationSpawner.palette == null && paletteEncoder != null)
            {
                vegetationSpawner.palette = paletteEncoder.palette;
            }

            if (vegetationSpawner.targetTerrain == null && terrainBuilder != null)
            {
                vegetationSpawner.targetTerrain = terrainBuilder.targetTerrain;
            }

            vegetationSpawner.Spawn(lastTerrainBiomeMap);
        }

        private void CacheTerrainTensor(Tensor<float> terrainTensor)
        {
            lastOutputData = terrainTensor.DownloadToArray();
            lastChannelCount = terrainTensor.shape[1];
            lastWidth = terrainTensor.shape[3];
            lastHeight = terrainTensor.shape[2];
        }

        private Tensor<float> BuildModelInputTensor(Texture2D biomeTexture, int inputResolution)
        {
            MicroBiomeMap modelInputMap = paletteEncoder.BuildMicroBiomeMap(biomeTexture, inputResolution, inputResolution);
            return paletteEncoder.BuildModelInputTensor(
                modelInputMap,
                Mathf.Max(1, inferenceController.modelInputChannels)
            );
        }

        private void ApplyCachedHeight()
        {
            heightPostprocessor.ConfigureOutputMirror(mirrorOutputWidth, mirrorOutputLength);
            int heightChannel = lastChannelCount > 1 ? Mathf.Clamp(generatedHeightChannel, 0, lastChannelCount - 1) : 0;
            float[,] heights = heightPostprocessor.TensorToHeights(
                lastOutputData,
                lastWidth,
                lastHeight,
                heightChannel,
                lastChannelCount);
            terrainBuilder.ApplyHeights(heights);
        }

        private void ApplyCachedTexture()
        {
            switch (textureMode)
            {
                case TerrainTextureMode.None:
                    lastAppliedTexture = null;
                    terrainBuilder.ApplyNoTexture();
                    return;
                case TerrainTextureMode.PaintedBiomeMap:
                    ApplyPaintedBiomeLayers();
                    return;
                case TerrainTextureMode.GeneratedTexture:
                    ApplyGeneratedTexture();
                    return;
                default:
                    throw new System.ArgumentOutOfRangeException(nameof(textureMode), textureMode, "Unsupported texture mode.");
            }
        }

        private void ApplyPaintedBiomeLayers()
        {
            if (lastBiomeTexture != null)
            {
                int inputResolution = Mathf.Max(1, inferenceController != null ? inferenceController.inputResolution : lastWidth);
                lastTerrainBiomeMap = BuildTerrainBiomeMap(lastBiomeTexture, ResolveTerrainBiomeMapResolution(inputResolution, lastBiomeTexture));
            }

            if (lastTerrainBiomeMap == null)
            {
                return;
            }

            lastAppliedTexture = null;
            terrainBuilder.ApplyBiomeAlphamaps(lastTerrainBiomeMap, paletteEncoder.palette);
        }

        private void ApplyGeneratedTexture()
        {
            if (texturePostprocessor == null)
            {
                Debug.LogWarning("TexturePostprocessor is missing. Falling back to painted biome layers.");
                ApplyPaintedBiomeLayers();
                return;
            }

            if (lastChannelCount < 3)
            {
                Debug.LogWarning("The current ONNX output has no generated RGB channels. Falling back to painted biome layers.");
                ApplyPaintedBiomeLayers();
                return;
            }

            texturePostprocessor.ConfigureOutputMirror(mirrorOutputWidth, mirrorOutputLength);
            Texture2D generatedTexture = texturePostprocessor.TensorToTexture(
                lastOutputData,
                lastWidth,
                lastHeight,
                lastChannelCount,
                generatedTextureRedChannel,
                generatedTextureGreenChannel,
                generatedTextureBlueChannel);
            lastGeneratedTexture = generatedTexture;
            lastAppliedTexture = generatedTexture;
            terrainBuilder.ApplySingleTexture(generatedTexture, true);
        }

        private MicroBiomeMap BuildTerrainBiomeMap(Texture2D biomeTexture, int inputResolution)
        {
            return paletteEncoder.BuildMicroBiomeMap(
                biomeTexture,
                inputResolution,
                inputResolution,
                mirrorOutputWidth,
                !mirrorOutputLength,
                false
            );
        }

        private int ResolveTerrainBiomeMapResolution(int inputResolution, Texture2D biomeTexture)
        {
            int terrainResolution = terrainBuilder != null ? terrainBuilder.alphamapResolution : inputResolution;
            int textureResolution = biomeTexture != null
                ? Mathf.Max(biomeTexture.width, biomeTexture.height)
                : inputResolution;
            return Mathf.Clamp(Mathf.Max(inputResolution, terrainResolution, textureResolution), 16, 4096);
        }

        private static MicroBiomeMap ExtractTile(MicroBiomeMap source, int startX, int startY, int tileSize)
        {
            int[] micro = new int[tileSize * tileSize];
            int[] macro = new int[tileSize * tileSize];
            for (int y = 0; y < tileSize; y++)
            {
                int sourceY = Mathf.Clamp(startY + y, 0, source.Height - 1);
                for (int x = 0; x < tileSize; x++)
                {
                    int sourceX = Mathf.Clamp(startX + x, 0, source.Width - 1);
                    int targetIndex = y * tileSize + x;
                    micro[targetIndex] = source.GetMicroBiomeIndex(sourceX, sourceY);
                    macro[targetIndex] = source.GetMacroClassIndex(sourceX, sourceY);
                }
            }

            return new MicroBiomeMap(tileSize, tileSize, micro, macro);
        }

        private static void AccumulateTile(
            float[] assembledData,
            float[] weights,
            float[] tileData,
            int channelCount,
            int tileSize,
            int assembledResolution,
            int startX,
            int startY,
            int overlap,
            int tileX,
            int tileY,
            int grid)
        {
            int assembledPlaneSize = assembledResolution * assembledResolution;
            int tilePlaneSize = tileSize * tileSize;
            for (int y = 0; y < tileSize; y++)
            {
                int globalY = startY + y;
                for (int x = 0; x < tileSize; x++)
                {
                    int globalX = startX + x;
                    int assembledIndex = globalY * assembledResolution + globalX;
                    int tileIndex = y * tileSize + x;
                    float weight = ComputeTileWeight(x, y, tileSize, overlap, tileX, tileY, grid);
                    weights[assembledIndex] += weight;

                    for (int channel = 0; channel < channelCount; channel++)
                    {
                        assembledData[channel * assembledPlaneSize + assembledIndex] +=
                            tileData[channel * tilePlaneSize + tileIndex] * weight;
                    }
                }
            }
        }

        private static float ComputeTileWeight(int x, int y, int tileSize, int overlap, int tileX, int tileY, int grid)
        {
            if (overlap <= 0)
            {
                return 1f;
            }

            float weightX = ComputeAxisWeight(x, tileSize, overlap, tileX > 0, tileX < grid - 1);
            float weightY = ComputeAxisWeight(y, tileSize, overlap, tileY > 0, tileY < grid - 1);
            return Mathf.Max(0.0001f, weightX * weightY);
        }

        private static float ComputeAxisWeight(int position, int tileSize, int overlap, bool hasPrevious, bool hasNext)
        {
            float weight = 1f;
            float denominator = Mathf.Max(1, overlap - 1);
            if (hasPrevious && position < overlap)
            {
                weight = Mathf.Min(weight, Mathf.SmoothStep(0f, 1f, position / denominator));
            }

            if (hasNext && position >= tileSize - overlap)
            {
                weight = Mathf.Min(weight, Mathf.SmoothStep(0f, 1f, (tileSize - 1 - position) / denominator));
            }

            return Mathf.Clamp01(weight);
        }

        private static void NormalizeAssembledTiles(float[] assembledData, float[] weights, int channelCount, int assembledResolution)
        {
            int planeSize = assembledResolution * assembledResolution;
            for (int index = 0; index < planeSize; index++)
            {
                float denominator = Mathf.Max(weights[index], 0.0001f);
                for (int channel = 0; channel < channelCount; channel++)
                {
                    int dataIndex = channel * planeSize + index;
                    assembledData[dataIndex] /= denominator;
                }
            }
        }

        private static int BuildTileSeed(int baseSeed, int tileX, int tileY, int grid)
        {
            unchecked
            {
                int mixed = Mathf.Max(0, baseSeed);
                mixed ^= (tileX + 1) * 73856093;
                mixed ^= (tileY + 1) * 19349663;
                mixed ^= Mathf.Max(1, grid) * 83492791;
                mixed ^= mixed >> 13;
                mixed *= 1274126177;
                mixed ^= mixed >> 16;
                return mixed == int.MinValue ? int.MaxValue : Mathf.Abs(mixed);
            }
        }
    }
}
