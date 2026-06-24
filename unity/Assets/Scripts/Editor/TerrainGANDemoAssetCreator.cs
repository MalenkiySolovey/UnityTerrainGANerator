using System.IO;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Biomes;
using UnityTerrainGANerator.Inference;
using UnityTerrainGANerator.Preview;
using UnityTerrainGANerator.TerrainGeneration;

namespace UnityTerrainGANerator.Editor
{
    public static class TerrainGANDemoAssetCreator
    {
        private const string RootFolder = "Assets/TerrainGAN";
        private const string BiomesFolder = RootFolder + "/Biomes";
        private const string MicroBiomesFolder = RootFolder + "/MicroBiomes";
        private const string LayersFolder = RootFolder + "/TerrainLayers";
        private const string PalettesFolder = RootFolder + "/Palettes";
        private const string ProfilesFolder = RootFolder + "/Profiles";
        private const string TexturesFolder = RootFolder + "/Textures";
        private const string ModelPath = "Assets/Models/spade_terrain_epoch90.onnx";

        private static readonly BiomeSpec[] Biomes =
        {
            new BiomeSpec("Water", 0, new Color32(17, 141, 215, 255), new Vector2(0f, 0.25f), 0f, 0f),
            new BiomeSpec("Sand", 1, new Color32(225, 227, 155, 255), new Vector2(0f, 0.35f), 0f, 0.1f),
            new BiomeSpec("Grass", 2, new Color32(127, 173, 123, 255), new Vector2(0.1f, 0.55f), 0.4f, 0.6f),
            new BiomeSpec("Dirt", 3, new Color32(185, 122, 87, 255), new Vector2(0.15f, 0.65f), 0.05f, 0.2f),
            new BiomeSpec("Snow", 4, new Color32(230, 200, 181, 255), new Vector2(0.7f, 1f), 0f, 0f),
            new BiomeSpec("Rock", 5, new Color32(150, 150, 150, 255), new Vector2(0.45f, 0.9f), 0f, 0.05f),
            new BiomeSpec("Mountain", 6, new Color32(193, 190, 175, 255), new Vector2(0.6f, 1f), 0f, 0f),
        };

        private static readonly MicroBiomeSpec[] MicroBiomes =
        {
            new MicroBiomeSpec("DeepWater", 100, 0, new Color32(8, 92, 180, 255), new Vector2(0f, 0.18f), new Vector2(0f, 12f), 0f, 0f, 0f, 28f),
            new MicroBiomeSpec("ShallowWater", 101, 0, new Color32(63, 169, 219, 255), new Vector2(0f, 0.24f), new Vector2(0f, 10f), 0f, 0f, 0f, 32f),
            new MicroBiomeSpec("Beach", 200, 1, new Color32(238, 221, 151, 255), new Vector2(0.02f, 0.32f), new Vector2(0f, 20f), 0f, 0.08f, 0f, 36f),
            new MicroBiomeSpec("Dunes", 201, 1, new Color32(215, 188, 111, 255), new Vector2(0.05f, 0.42f), new Vector2(0f, 28f), 0f, 0.04f, 0.02f, 45f),
            new MicroBiomeSpec("Meadow", 300, 2, new Color32(96, 183, 98, 255), new Vector2(0.08f, 0.58f), new Vector2(0f, 28f), 0.16f, 0.55f, 0.01f, 38f),
            new MicroBiomeSpec("Forest", 301, 2, new Color32(45, 128, 73, 255), new Vector2(0.12f, 0.65f), new Vector2(0f, 35f), 0.75f, 0.45f, 0.02f, 42f),
            new MicroBiomeSpec("SparseGrass", 302, 2, new Color32(147, 175, 83, 255), new Vector2(0.08f, 0.6f), new Vector2(0f, 32f), 0.08f, 0.25f, 0.02f, 52f),
            new MicroBiomeSpec("DryDirt", 400, 3, new Color32(164, 103, 67, 255), new Vector2(0.12f, 0.68f), new Vector2(0f, 40f), 0.03f, 0.12f, 0.03f, 48f),
            new MicroBiomeSpec("Mud", 401, 3, new Color32(96, 73, 54, 255), new Vector2(0.04f, 0.38f), new Vector2(0f, 22f), 0.02f, 0.08f, 0f, 34f),
            new MicroBiomeSpec("FreshSnow", 500, 4, new Color32(238, 238, 231, 255), new Vector2(0.62f, 1f), new Vector2(0f, 38f), 0f, 0.02f, 0f, 30f),
            new MicroBiomeSpec("Ice", 501, 4, new Color32(176, 218, 229, 255), new Vector2(0.55f, 1f), new Vector2(0f, 24f), 0f, 0f, 0f, 26f),
            new MicroBiomeSpec("Gravel", 600, 5, new Color32(111, 111, 107, 255), new Vector2(0.25f, 0.88f), new Vector2(5f, 52f), 0f, 0.08f, 0.25f, 56f),
            new MicroBiomeSpec("BareRock", 601, 5, new Color32(78, 82, 82, 255), new Vector2(0.35f, 0.95f), new Vector2(12f, 70f), 0f, 0.02f, 0.18f, 62f),
            new MicroBiomeSpec("AlpineRock", 700, 6, new Color32(171, 169, 158, 255), new Vector2(0.5f, 1f), new Vector2(15f, 75f), 0f, 0.02f, 0.2f, 58f),
            new MicroBiomeSpec("HighMountain", 701, 6, new Color32(216, 213, 197, 255), new Vector2(0.68f, 1f), new Vector2(18f, 82f), 0f, 0f, 0.08f, 44f),
        };

        [MenuItem("Tools/Terrain GAN/Create Demo Assets")]
        public static void CreateDemoAssetsFromMenu()
        {
            CreateDemoAssets();
        }

        [MenuItem("Tools/Terrain GAN/Create Default Micro Biomes")]
        public static void CreateDefaultMicroBiomeAssetsFromMenu()
        {
            CreateDefaultMicroBiomeAssets();
        }

        public static void CreateDemoAssets()
        {
            EnsureFolders();
            BiomeDefinition[] definitions = CreateBiomeDefinitions();
            BiomePalette palette = CreatePalette(definitions);
            Texture2D texture = CreateDemoBiomeTexture();
            TerrainGANPreviewProfile profile = CreatePreviewProfile(palette, texture);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Selection.activeObject = profile;
            Debug.Log($"Terrain GAN demo assets are ready. Profile: {AssetDatabase.GetAssetPath(profile)}, palette: {AssetDatabase.GetAssetPath(palette)}, texture: {AssetDatabase.GetAssetPath(texture)}");
        }

        public static void CreateDefaultMicroBiomeAssets()
        {
            EnsureFolders();
            BiomeDefinition[] macroDefinitions = CreateBiomeDefinitions();
            BiomeDefinition[] microDefinitions = CreateMicroBiomeDefinitions(macroDefinitions);
            BiomeDefinition[] paletteDefinitions = MergeDefinitions(macroDefinitions, microDefinitions);
            BiomePalette palette = CreatePalette(paletteDefinitions);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Selection.activeObject = palette;
            Debug.Log($"Terrain GAN micro-biome assets are ready. Palette: {AssetDatabase.GetAssetPath(palette)}, micro-biomes: {microDefinitions.Length}");
        }

        private static void EnsureFolders()
        {
            Directory.CreateDirectory(RootFolder);
            Directory.CreateDirectory(BiomesFolder);
            Directory.CreateDirectory(MicroBiomesFolder);
            Directory.CreateDirectory(LayersFolder);
            Directory.CreateDirectory(PalettesFolder);
            Directory.CreateDirectory(ProfilesFolder);
            Directory.CreateDirectory(TexturesFolder);
        }

        private static BiomeDefinition[] CreateBiomeDefinitions()
        {
            BiomeDefinition[] definitions = new BiomeDefinition[Biomes.Length];
            for (int i = 0; i < Biomes.Length; i++)
            {
                BiomeSpec spec = Biomes[i];
                string path = $"{BiomesFolder}/Biome_{spec.Name}.asset";
                BiomeDefinition definition = AssetDatabase.LoadAssetAtPath<BiomeDefinition>(path);
                if (definition == null)
                {
                    definition = ScriptableObject.CreateInstance<BiomeDefinition>();
                    AssetDatabase.CreateAsset(definition, path);
                }

                definition.biomeName = spec.Name.ToLowerInvariant();
                definition.microBiomeId = spec.ClassIndex;
                definition.macroClassIndex = spec.ClassIndex;
                definition.inputColor = spec.Color;
                definition.macroInputColor = spec.Color;
                definition.heightRange = spec.HeightRange;
                definition.slopeRange = new Vector2(0f, 90f);
                definition.terrainLayers = new[] { CreateTerrainLayer(spec) };
                definition.terrainLayerWeights = new[] { 1f };
                definition.treeDensity = spec.TreeDensity;
                definition.detailDensity = spec.DetailDensity;
                definition.rockDensity = spec.ClassIndex == 5 || spec.ClassIndex == 6 ? 0.08f : 0f;
                definition.noiseScale = 35f;
                definition.treeScaleRange = new Vector2(0.85f, 1.25f);
                definition.objectScaleRange = new Vector2(0.75f, 1.35f);
                definition.placementJitter = 0.85f;
                definition.alignObjectsToTerrainNormal = true;
                definition.normalAlignment = 0.85f;
                definition.seedOffset = spec.ClassIndex * 101;

                EditorUtility.SetDirty(definition);
                definitions[i] = definition;
            }

            return definitions;
        }

        private static BiomeDefinition[] CreateMicroBiomeDefinitions(BiomeDefinition[] macroDefinitions)
        {
            BiomeDefinition[] definitions = new BiomeDefinition[MicroBiomes.Length];
            for (int i = 0; i < MicroBiomes.Length; i++)
            {
                MicroBiomeSpec spec = MicroBiomes[i];
                string path = $"{MicroBiomesFolder}/MicroBiome_{spec.Name}.asset";
                BiomeDefinition definition = AssetDatabase.LoadAssetAtPath<BiomeDefinition>(path);
                if (definition == null)
                {
                    definition = ScriptableObject.CreateInstance<BiomeDefinition>();
                    AssetDatabase.CreateAsset(definition, path);
                }

                BiomeDefinition macroDefinition = macroDefinitions[Mathf.Clamp(spec.MacroClassIndex, 0, macroDefinitions.Length - 1)];
                definition.biomeName = spec.Name.ToLowerInvariant();
                definition.microBiomeId = spec.MicroBiomeId;
                definition.macroClassIndex = spec.MacroClassIndex;
                definition.inputColor = spec.Color;
                definition.macroInputColor = macroDefinition.MacroColor;
                definition.heightRange = spec.HeightRange;
                definition.slopeRange = spec.SlopeRange;
                definition.terrainLayers = new[] { CreateTerrainLayer(spec) };
                definition.terrainLayerWeights = new[] { 1f };
                definition.treeDensity = spec.TreeDensity;
                definition.detailDensity = spec.DetailDensity;
                definition.rockDensity = spec.RockDensity;
                definition.noiseScale = spec.NoiseScale;
                definition.treeScaleRange = new Vector2(0.75f, 1.35f);
                definition.objectScaleRange = new Vector2(0.65f, 1.45f);
                definition.placementJitter = 0.9f;
                definition.alignObjectsToTerrainNormal = true;
                definition.normalAlignment = 0.85f;
                definition.seedOffset = spec.MicroBiomeId * 101;

                EditorUtility.SetDirty(definition);
                definitions[i] = definition;
            }

            return definitions;
        }

        private static TerrainLayer CreateTerrainLayer(BiomeSpec spec)
        {
            Texture2D diffuseTexture = CreateSolidTexture(spec);
            string path = $"{LayersFolder}/Layer_{spec.Name}.terrainlayer";
            TerrainLayer layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(path);
            if (layer == null)
            {
                layer = new TerrainLayer();
                AssetDatabase.CreateAsset(layer, path);
            }

            layer.diffuseTexture = diffuseTexture;
            layer.tileSize = new Vector2(18f, 18f);
            layer.tileOffset = Vector2.zero;
            layer.metallic = 0f;
            layer.smoothness = 0f;
            layer.smoothnessSource = TerrainLayerSmoothnessSource.Constant;
            EditorUtility.SetDirty(layer);
            return layer;
        }

        private static TerrainLayer CreateTerrainLayer(MicroBiomeSpec spec)
        {
            Texture2D diffuseTexture = CreateSolidTexture(spec);
            string path = $"{LayersFolder}/Layer_{spec.Name}.terrainlayer";
            TerrainLayer layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(path);
            if (layer == null)
            {
                layer = new TerrainLayer();
                AssetDatabase.CreateAsset(layer, path);
            }

            layer.diffuseTexture = diffuseTexture;
            layer.tileSize = new Vector2(14f, 14f);
            layer.tileOffset = Vector2.zero;
            layer.metallic = 0f;
            layer.smoothness = 0f;
            layer.smoothnessSource = TerrainLayerSmoothnessSource.Constant;
            EditorUtility.SetDirty(layer);
            return layer;
        }

        private static Texture2D CreateSolidTexture(BiomeSpec spec)
        {
            string path = $"{TexturesFolder}/terrain_{spec.Name.ToLowerInvariant()}.png";
            Texture2D texture = new Texture2D(4, 4, TextureFormat.RGBA32, false);
            Color32[] pixels = new Color32[16];

            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = spec.Color;
            }

            texture.SetPixels32(pixels);
            texture.Apply();
            File.WriteAllBytes(path, texture.EncodeToPNG());
            Object.DestroyImmediate(texture);

            AssetDatabase.ImportAsset(path);
            TextureImporter importer = AssetImporter.GetAtPath(path) as TextureImporter;
            if (importer != null)
            {
                importer.isReadable = true;
                importer.mipmapEnabled = true;
                importer.filterMode = FilterMode.Bilinear;
                importer.textureCompression = TextureImporterCompression.Uncompressed;
                importer.SaveAndReimport();
            }

            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        private static Texture2D CreateSolidTexture(MicroBiomeSpec spec)
        {
            string path = $"{TexturesFolder}/terrain_micro_{spec.Name.ToLowerInvariant()}.png";
            Texture2D texture = new Texture2D(4, 4, TextureFormat.RGBA32, false);
            Color32[] pixels = new Color32[16];

            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = spec.Color;
            }

            texture.SetPixels32(pixels);
            texture.Apply();
            File.WriteAllBytes(path, texture.EncodeToPNG());
            Object.DestroyImmediate(texture);

            AssetDatabase.ImportAsset(path);
            TextureImporter importer = AssetImporter.GetAtPath(path) as TextureImporter;
            if (importer != null)
            {
                importer.isReadable = true;
                importer.mipmapEnabled = true;
                importer.filterMode = FilterMode.Bilinear;
                importer.textureCompression = TextureImporterCompression.Uncompressed;
                importer.SaveAndReimport();
            }

            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        private static BiomePalette CreatePalette(BiomeDefinition[] definitions)
        {
            const string path = PalettesFolder + "/Dataset_BiomePalette.asset";
            BiomePalette palette = AssetDatabase.LoadAssetAtPath<BiomePalette>(path);
            if (palette == null)
            {
                palette = ScriptableObject.CreateInstance<BiomePalette>();
                AssetDatabase.CreateAsset(palette, path);
            }

            palette.biomes = definitions;
            palette.macroClassCount = 7;
            palette.colorTolerance = 8;
            palette.fallbackMacroClassIndex = 2;
            EditorUtility.SetDirty(palette);
            return palette;
        }

        private static BiomeDefinition[] MergeDefinitions(BiomeDefinition[] macroDefinitions, BiomeDefinition[] microDefinitions)
        {
            BiomeDefinition[] merged = new BiomeDefinition[macroDefinitions.Length + microDefinitions.Length];
            for (int i = 0; i < macroDefinitions.Length; i++)
            {
                merged[i] = macroDefinitions[i];
            }

            for (int i = 0; i < microDefinitions.Length; i++)
            {
                merged[macroDefinitions.Length + i] = microDefinitions[i];
            }

            return merged;
        }

        private static TerrainGANPreviewProfile CreatePreviewProfile(BiomePalette palette, Texture2D biomeTexture)
        {
            const string path = ProfilesFolder + "/DemoPreviewProfile.asset";
            TerrainGANPreviewProfile profile = AssetDatabase.LoadAssetAtPath<TerrainGANPreviewProfile>(path);
            if (profile == null)
            {
                profile = ScriptableObject.CreateInstance<TerrainGANPreviewProfile>();
                AssetDatabase.CreateAsset(profile, path);
            }

            profile.palette = palette;
            profile.biomeTexture = biomeTexture;
            profile.modelAsset = LoadDefaultModelAsset();
            profile.backendType = BackendType.CPU;
            profile.biomeInputName = "biome_onehot";
            profile.noiseInputName = "noise";
            profile.terrainOutputName = "terrain";
            profile.useNoiseInputs = true;
            profile.noiseSeed = 12345;
            profile.noiseStrength = 1f;
            profile.noiseDistribution = LatentNoiseDistribution.Uniform01;
            profile.noiseDim = 16;
            profile.useTrainingNoiseInputs = true;
            profile.trainingNoiseInputPrefix = "train_noise_";
            profile.trainingNoiseStrength = 1f;
            profile.inputResolution = 256;
            profile.macroClassCount = 7;
            profile.modelInputChannels = 11;
            profile.textureMode = TerrainTextureMode.GeneratedTexture;
            profile.generatedTextureRedChannel = 0;
            profile.generatedTextureGreenChannel = 1;
            profile.generatedTextureBlueChannel = 2;
            profile.generatedHeightChannel = 3;
            profile.textureSaturation = 1f;
            profile.textureBrightness = 1f;
            profile.textureContrast = 1f;
            profile.textureGamma = 1f;
            profile.textureTint = Color.white;
            profile.invertOutputWidth = false;
            profile.invertOutputLength = false;
            profile.terrainSize = new Vector3(1000f, 220f, 1000f);
            profile.heightmapResolution = 257;
            profile.alphamapResolution = 256;
            profile.useDelayedLod = false;
            profile.paintTerrainLayers = false;
            profile.useTiledGeneration = false;
            profile.tileGrid = 1;
            profile.tileOverlapPixels = 32;
            profile.usePerTileVariation = true;
            profile.clamp01 = true;
            profile.resize256To257 = true;
            profile.remapWithPercentiles = true;
            profile.lowerPercentile = 0.02f;
            profile.upperPercentile = 0.98f;
            profile.outputMin = 0.02f;
            profile.outputMax = 0.42f;
            profile.smoothingIterations = 0;
            profile.smoothingStrength = 0.65f;
            EditorUtility.SetDirty(profile);
            return profile;
        }

        private static ModelAsset LoadDefaultModelAsset()
        {
            string[] preferredPaths =
            {
                ModelPath,
                "Assets/Models/spade_terrain_epoch128.onnx",
                "Assets/Models/spade_terrain_epoch116.onnx",
                "Assets/Models/spade_terrain_rgb_height_v12_4_feature_supervised_256.onnx"
            };

            foreach (string path in preferredPaths)
            {
                ModelAsset model = AssetDatabase.LoadAssetAtPath<ModelAsset>(path);
                if (model != null)
                {
                    return model;
                }
            }

            string[] allModels = AssetDatabase.FindAssets("t:ModelAsset", new[] { "Assets/Models" });
            foreach (string guid in allModels)
            {
                ModelAsset model = AssetDatabase.LoadAssetAtPath<ModelAsset>(AssetDatabase.GUIDToAssetPath(guid));
                if (model != null)
                {
                    return model;
                }
            }

            return null;
        }

        private static Texture2D CreateDemoBiomeTexture()
        {
            const int size = 256;
            const string path = TexturesFolder + "/demo_biome_map.png";
            Texture2D texture = new Texture2D(size, size, TextureFormat.RGBA32, false);
            Color32[] pixels = new Color32[size * size];

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float nx = (x - size * 0.5f) / (size * 0.5f);
                    float ny = (y - size * 0.5f) / (size * 0.5f);
                    float radius = Mathf.Sqrt(nx * nx + ny * ny);
                    Color32 color = Biomes[2].Color;

                    if (radius > 0.88f)
                    {
                        color = Biomes[0].Color;
                    }
                    else if (radius > 0.78f)
                    {
                        color = Biomes[1].Color;
                    }
                    else if (radius < 0.16f)
                    {
                        color = Biomes[4].Color;
                    }
                    else if (radius < 0.3f)
                    {
                        color = Biomes[6].Color;
                    }
                    else if (radius < 0.46f)
                    {
                        color = Biomes[5].Color;
                    }
                    else if (x < 92 && y > 120)
                    {
                        color = Biomes[3].Color;
                    }

                    pixels[y * size + x] = color;
                }
            }

            texture.SetPixels32(pixels);
            texture.Apply();
            File.WriteAllBytes(path, texture.EncodeToPNG());
            Object.DestroyImmediate(texture);

            AssetDatabase.ImportAsset(path);
            TextureImporter importer = AssetImporter.GetAtPath(path) as TextureImporter;
            if (importer != null)
            {
                importer.isReadable = true;
                importer.mipmapEnabled = false;
                importer.filterMode = FilterMode.Point;
                importer.textureCompression = TextureImporterCompression.Uncompressed;
                importer.SaveAndReimport();
            }

            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        private readonly struct BiomeSpec
        {
            public BiomeSpec(string name, int classIndex, Color32 color, Vector2 heightRange, float treeDensity, float detailDensity)
            {
                Name = name;
                ClassIndex = classIndex;
                Color = color;
                HeightRange = heightRange;
                TreeDensity = treeDensity;
                DetailDensity = detailDensity;
            }

            public string Name { get; }
            public int ClassIndex { get; }
            public Color32 Color { get; }
            public Vector2 HeightRange { get; }
            public float TreeDensity { get; }
            public float DetailDensity { get; }
        }

        private readonly struct MicroBiomeSpec
        {
            public MicroBiomeSpec(
                string name,
                int microBiomeId,
                int macroClassIndex,
                Color32 color,
                Vector2 heightRange,
                Vector2 slopeRange,
                float treeDensity,
                float detailDensity,
                float rockDensity,
                float noiseScale)
            {
                Name = name;
                MicroBiomeId = microBiomeId;
                MacroClassIndex = macroClassIndex;
                Color = color;
                HeightRange = heightRange;
                SlopeRange = slopeRange;
                TreeDensity = treeDensity;
                DetailDensity = detailDensity;
                RockDensity = rockDensity;
                NoiseScale = noiseScale;
            }

            public string Name { get; }
            public int MicroBiomeId { get; }
            public int MacroClassIndex { get; }
            public Color32 Color { get; }
            public Vector2 HeightRange { get; }
            public Vector2 SlopeRange { get; }
            public float TreeDensity { get; }
            public float DetailDensity { get; }
            public float RockDensity { get; }
            public float NoiseScale { get; }
        }
    }
}
