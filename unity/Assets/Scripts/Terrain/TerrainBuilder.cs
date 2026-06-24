using System.Collections.Generic;
#if UNITY_EDITOR
using System.IO;
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.Rendering;
using UnityTerrainGANerator.Biomes;

namespace UnityTerrainGANerator.TerrainGeneration
{
    public sealed class TerrainBuilder : MonoBehaviour
    {
        public Terrain targetTerrain;
        public int heightmapResolution = 257;
        public int alphamapResolution = 256;
        public Vector3 terrainSize = new Vector3(1000f, 220f, 1000f);
        public bool useDelayedLod = false;
        public Color noTextureColor = new Color(0.5f, 0.5f, 0.5f, 1f);

        private TerrainLayer runtimeTextureLayer;
        private Texture2D runtimeSolidTexture;
        private Texture2D runtimeOwnedTexture;

#if UNITY_EDITOR
        private const string PersistentFolder = "Assets/TerrainGAN/Persistent";
        private const string PersistentGeneratedTexturePath = PersistentFolder + "/generated_onnx_texture.png";
        private const string PersistentGeneratedLayerPath = PersistentFolder + "/Generated_ONNX_Texture.terrainlayer";
        private const string PersistentSolidTexturePath = PersistentFolder + "/solid_terrain_texture.png";
        private const string PersistentSolidLayerPath = PersistentFolder + "/Solid_Terrain_Texture.terrainlayer";
#endif

        public void ApplyHeights(float[,] heights)
        {
            if (targetTerrain == null)
            {
                targetTerrain = GetComponent<Terrain>();
            }

            if (targetTerrain == null)
            {
                throw new System.InvalidOperationException("Target Terrain is not assigned.");
            }

            ConfigureTerrainRendering();
            TerrainData data = targetTerrain.terrainData;
            data.heightmapResolution = heights.GetLength(0);
            data.size = terrainSize;

            if (useDelayedLod)
            {
                data.SetHeightsDelayLOD(0, 0, heights);
                data.SyncHeightmap();
            }
            else
            {
                data.SetHeights(0, 0, heights);
            }

            targetTerrain.Flush();
            MarkTerrainDirty(data);
        }

        public void ApplyBiomeAlphamaps(MicroBiomeMap biomeMap, BiomePalette palette)
        {
            if (biomeMap == null)
            {
                throw new System.ArgumentNullException(nameof(biomeMap));
            }

            if (palette == null)
            {
                throw new System.ArgumentNullException(nameof(palette));
            }

            TerrainData data = GetTerrainData();
            List<TerrainLayer> layers = new List<TerrainLayer>();
            Dictionary<TerrainLayer, int> layerToIndex = new Dictionary<TerrainLayer, int>();
            Dictionary<int, List<LayerWeight>> microBiomeLayers = new Dictionary<int, List<LayerWeight>>();
            Dictionary<int, List<LayerWeight>> macroBiomeFallbackLayers = new Dictionary<int, List<LayerWeight>>();

            if (palette.biomes != null)
            {
                foreach (BiomeDefinition biome in palette.biomes)
                {
                    List<LayerWeight> weights = BuildLayerWeights(biome, layers, layerToIndex);
                    if (weights == null || weights.Count == 0)
                    {
                        continue;
                    }

                    microBiomeLayers[biome.microBiomeId] = weights;
                    int macroClass = Mathf.Clamp(biome.macroClassIndex, 0, Mathf.Max(1, palette.macroClassCount) - 1);
                    if (!macroBiomeFallbackLayers.ContainsKey(macroClass) || biome.microBiomeId == macroClass)
                    {
                        macroBiomeFallbackLayers[macroClass] = weights;
                    }
                }
            }

            if (layers.Count == 0)
            {
                return;
            }

            int fallbackMacroClass = Mathf.Clamp(palette.fallbackMacroClassIndex, 0, Mathf.Max(1, palette.macroClassCount) - 1);
            List<LayerWeight> fallbackWeights = macroBiomeFallbackLayers.TryGetValue(fallbackMacroClass, out List<LayerWeight> resolvedFallback)
                ? resolvedFallback
                : new List<LayerWeight> { new LayerWeight(0, 1f) };

            DestroyRuntimeObject(runtimeTextureLayer);
            runtimeTextureLayer = null;
            runtimeOwnedTexture = null;

            for (int index = 0; index < layers.Count; index++)
            {
                ConfigureMatteLayer(layers[index]);
            }

            data.terrainLayers = layers.ToArray();
            data.alphamapResolution = Mathf.Max(16, alphamapResolution);

            int width = data.alphamapWidth;
            int height = data.alphamapHeight;
            float[,,] alphamaps = new float[width, height, layers.Count];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float u = width == 1 ? 0f : (float)x / (width - 1);
                    float v = height == 1 ? 0f : (float)y / (height - 1);
                    int mapX = Mathf.Clamp(Mathf.RoundToInt(u * (biomeMap.Width - 1)), 0, biomeMap.Width - 1);
                    int mapY = Mathf.Clamp(Mathf.RoundToInt(v * (biomeMap.Height - 1)), 0, biomeMap.Height - 1);
                    int microBiome = biomeMap.GetMicroBiomeIndex(mapX, mapY);
                    int macroClass = Mathf.Clamp(biomeMap.GetMacroClassIndex(mapX, mapY), 0, Mathf.Max(1, palette.macroClassCount) - 1);
                    List<LayerWeight> weights = ResolveLayerWeights(microBiome, macroClass, microBiomeLayers, macroBiomeFallbackLayers, fallbackWeights);

                    foreach (LayerWeight weight in weights)
                    {
                        alphamaps[x, y, weight.LayerIndex] = weight.Weight;
                    }
                }
            }

            data.SetAlphamaps(0, 0, alphamaps);
            MarkTerrainDirty(data);
        }

        public void ApplyAlphamaps(float[,,] alphamaps)
        {
            TerrainData data = GetTerrainData();
            data.SetAlphamaps(0, 0, alphamaps);
            MarkTerrainDirty(data);
        }

        public void ApplySingleTexture(Texture2D texture)
        {
            ApplySingleTexture(texture, false);
        }

        public void ApplySingleTexture(Texture2D texture, bool ownsTexture)
        {
            if (texture == null)
            {
                throw new System.ArgumentNullException(nameof(texture));
            }

            TerrainData data = GetTerrainData();
            TerrainLayer previousLayer = runtimeTextureLayer;
            Texture2D previousOwnedTexture = runtimeOwnedTexture;

            TerrainLayer nextLayer = CreateSingleTextureLayer(texture, ownsTexture, data);

            data.terrainLayers = new TerrainLayer[0];
            runtimeTextureLayer = nextLayer;
            runtimeOwnedTexture = ownsTexture ? texture : null;
            data.terrainLayers = new[] { runtimeTextureLayer };
            ApplyFullLayerAlphamap(data);
            targetTerrain.Flush();
            MarkTerrainDirty(data);

            DestroyRuntimeObject(previousLayer);
            if (previousOwnedTexture != null && previousOwnedTexture != texture && previousOwnedTexture != runtimeSolidTexture)
            {
                DestroyRuntimeObject(previousOwnedTexture);
            }
        }

        public void ApplyNoTexture()
        {
            TerrainData data = GetTerrainData();
            if (runtimeSolidTexture == null)
            {
                runtimeSolidTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false, false);
                runtimeSolidTexture.name = "Runtime Solid Terrain Texture";
                runtimeSolidTexture.wrapMode = TextureWrapMode.Clamp;
                runtimeSolidTexture.filterMode = FilterMode.Point;
            }

            Color[] pixels =
            {
                noTextureColor,
                noTextureColor,
                noTextureColor,
                noTextureColor
            };
            runtimeSolidTexture.SetPixels(pixels);
            runtimeSolidTexture.Apply(false, false);
            ApplySingleTexture(runtimeSolidTexture, false);
        }

        private TerrainData GetTerrainData()
        {
            if (targetTerrain == null)
            {
                targetTerrain = GetComponent<Terrain>();
            }

            if (targetTerrain == null)
            {
                throw new System.InvalidOperationException("Target Terrain is not assigned.");
            }

            ConfigureTerrainRendering();
            return targetTerrain.terrainData;
        }

        private void ConfigureTerrainRendering()
        {
            if (targetTerrain == null)
            {
                return;
            }

            targetTerrain.reflectionProbeUsage = ReflectionProbeUsage.Off;
        }

        private void MarkTerrainDirty(TerrainData data)
        {
#if UNITY_EDITOR
            if (Application.isPlaying)
            {
                return;
            }

            if (data != null)
            {
                EditorUtility.SetDirty(data);
            }

            if (targetTerrain != null)
            {
                EditorUtility.SetDirty(targetTerrain);
            }
#endif
        }

        private static void ApplyFullLayerAlphamap(TerrainData data)
        {
            int width = data.alphamapWidth;
            int height = data.alphamapHeight;
            float[,,] alphamaps = new float[height, width, 1];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    alphamaps[y, x, 0] = 1f;
                }
            }

            data.SetAlphamaps(0, 0, alphamaps);
        }

        private static void DestroyRuntimeObject(Object runtimeObject)
        {
            if (runtimeObject == null)
            {
                return;
            }

#if UNITY_EDITOR
            if (!string.IsNullOrEmpty(AssetDatabase.GetAssetPath(runtimeObject)))
            {
                return;
            }
#endif

            if (Application.isPlaying)
            {
                Object.Destroy(runtimeObject);
            }
            else
            {
                Object.DestroyImmediate(runtimeObject);
            }
        }

        private TerrainLayer CreateSingleTextureLayer(Texture2D texture, bool ownsTexture, TerrainData data)
        {
#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                return CreatePersistentSingleTextureLayer(texture, ownsTexture, data);
            }
#endif

            TerrainLayer layer = new TerrainLayer();
            layer.name = $"Runtime Layer - {texture.name}";
            layer.diffuseTexture = texture;
            layer.tileSize = new Vector2(data.size.x, data.size.z);
            ConfigureMatteLayer(layer);
            return layer;
        }

        private static void ConfigureMatteLayer(TerrainLayer layer)
        {
            if (layer == null)
            {
                return;
            }

            layer.metallic = 0f;
            layer.smoothness = 0f;
            layer.smoothnessSource = TerrainLayerSmoothnessSource.Constant;
            layer.specular = Color.black;
        }

#if UNITY_EDITOR
        private static TerrainLayer CreatePersistentSingleTextureLayer(Texture2D texture, bool ownsTexture, TerrainData data)
        {
            EnsurePersistentFolder();
            bool solidTexture = !ownsTexture && texture.name.Contains("Solid");
            string texturePath = solidTexture ? PersistentSolidTexturePath : PersistentGeneratedTexturePath;
            string layerPath = solidTexture ? PersistentSolidLayerPath : PersistentGeneratedLayerPath;
            Texture2D persistentTexture = SaveTextureAsset(texture, texturePath);

            TerrainLayer layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(layerPath);
            if (layer == null)
            {
                layer = new TerrainLayer();
                AssetDatabase.CreateAsset(layer, layerPath);
            }

            layer.name = Path.GetFileNameWithoutExtension(layerPath);
            layer.diffuseTexture = persistentTexture != null ? persistentTexture : texture;
            layer.tileSize = new Vector2(data.size.x, data.size.z);
            layer.tileOffset = Vector2.zero;
            ConfigureMatteLayer(layer);
            EditorUtility.SetDirty(layer);
            AssetDatabase.SaveAssets();
            return layer;
        }

        private static Texture2D SaveTextureAsset(Texture2D texture, string path)
        {
            if (texture == null)
            {
                return null;
            }

            EnsurePersistentFolder();
            File.WriteAllBytes(path, texture.EncodeToPNG());
            AssetDatabase.ImportAsset(path);

            TextureImporter importer = AssetImporter.GetAtPath(path) as TextureImporter;
            if (importer != null)
            {
                importer.textureType = TextureImporterType.Default;
                importer.sRGBTexture = true;
                importer.alphaSource = TextureImporterAlphaSource.None;
                importer.mipmapEnabled = true;
                importer.isReadable = false;
                importer.wrapMode = TextureWrapMode.Clamp;
                importer.filterMode = FilterMode.Bilinear;
                importer.textureCompression = TextureImporterCompression.Uncompressed;
                importer.SaveAndReimport();
            }

            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        private static void EnsurePersistentFolder()
        {
            if (!AssetDatabase.IsValidFolder("Assets/TerrainGAN"))
            {
                AssetDatabase.CreateFolder("Assets", "TerrainGAN");
            }

            if (!AssetDatabase.IsValidFolder(PersistentFolder))
            {
                AssetDatabase.CreateFolder("Assets/TerrainGAN", "Persistent");
            }
        }
#endif

        private static List<LayerWeight> BuildLayerWeights(
            BiomeDefinition biome,
            List<TerrainLayer> layers,
            Dictionary<TerrainLayer, int> layerToIndex)
        {
            if (biome == null || biome.terrainLayers == null || biome.terrainLayers.Length == 0)
            {
                return null;
            }

            List<LayerWeight> weights = new List<LayerWeight>();
            float totalWeight = 0f;
            for (int i = 0; i < biome.terrainLayers.Length; i++)
            {
                TerrainLayer layer = biome.terrainLayers[i];
                if (layer == null)
                {
                    continue;
                }

                if (!layerToIndex.TryGetValue(layer, out int layerIndex))
                {
                    layerIndex = layers.Count;
                    layers.Add(layer);
                    layerToIndex.Add(layer, layerIndex);
                }

                float weight = biome.GetTerrainLayerWeight(i);
                if (weight <= 0f)
                {
                    continue;
                }

                weights.Add(new LayerWeight(layerIndex, weight));
                totalWeight += weight;
            }

            if (weights.Count == 0)
            {
                return null;
            }

            for (int i = 0; i < weights.Count; i++)
            {
                LayerWeight weight = weights[i];
                weights[i] = new LayerWeight(weight.LayerIndex, weight.Weight / totalWeight);
            }

            return weights;
        }

        private static List<LayerWeight> ResolveLayerWeights(
            int microBiome,
            int macroClass,
            Dictionary<int, List<LayerWeight>> microBiomeLayers,
            Dictionary<int, List<LayerWeight>> macroBiomeFallbackLayers,
            List<LayerWeight> fallbackWeights)
        {
            if (microBiomeLayers.TryGetValue(microBiome, out List<LayerWeight> microWeights))
            {
                return microWeights;
            }

            if (macroBiomeFallbackLayers.TryGetValue(macroClass, out List<LayerWeight> macroWeights))
            {
                return macroWeights;
            }

            return fallbackWeights;
        }

        private readonly struct LayerWeight
        {
            public LayerWeight(int layerIndex, float weight)
            {
                LayerIndex = layerIndex;
                Weight = weight;
            }

            public int LayerIndex { get; }
            public float Weight { get; }
        }
    }
}
