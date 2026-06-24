using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityTerrainGANerator.Biomes;
using UnityTerrainGANerator.Demo;
using UnityTerrainGANerator.Inference;
using UnityTerrainGANerator.Preview;
using UnityTerrainGANerator.TerrainGeneration;

namespace UnityTerrainGANerator.Editor
{
    public static class TerrainGANSceneSetup
    {
        private const string RuntimeObjectName = "TerrainGANRuntime";
        private const string TerrainObjectName = "TerrainGAN_Terrain";
        private const string PalettePath = "Assets/TerrainGAN/Palettes/Dataset_BiomePalette.asset";
        private const string DemoTexturePath = "Assets/TerrainGAN/Textures/demo_biome_map.png";
        private const string ProfilePath = "Assets/TerrainGAN/Profiles/DemoPreviewProfile.asset";
        private const string TerrainDataFolder = "Assets/TerrainGAN/Terrain";
        private const string TerrainDataPath = TerrainDataFolder + "/DemoTerrainData.asset";

        public static void SetupDemoScene()
        {
            SetupDemoSceneAndGetController();
        }

        public static ExampleSceneController SetupDemoSceneAndGetController(Terrain preferredTerrain = null, TerrainGANPreviewProfile preferredProfile = null)
        {
            if (!DefaultSceneAssetsExist())
            {
                TerrainGANDemoAssetCreator.CreateDemoAssets();
            }

            TerrainGANPreviewProfile profile = preferredProfile != null
                ? preferredProfile
                : AssetDatabase.LoadAssetAtPath<TerrainGANPreviewProfile>(ProfilePath);
            BiomePalette palette = profile != null && profile.palette != null
                ? profile.palette
                : AssetDatabase.LoadAssetAtPath<BiomePalette>(PalettePath);
            Texture2D demoTexture = profile != null && profile.biomeTexture != null
                ? profile.biomeTexture
                : AssetDatabase.LoadAssetAtPath<Texture2D>(DemoTexturePath);

            if (palette == null || demoTexture == null || profile == null)
            {
                Debug.LogError("Terrain GAN setup failed: preview profile, palette, or biome texture was not found.");
                return null;
            }

            Terrain terrain = FindOrCreateTerrain(preferredTerrain);
            GameObject runtimeObject = GameObject.Find(RuntimeObjectName);
            if (runtimeObject == null)
            {
                runtimeObject = new GameObject(RuntimeObjectName);
            }

            PaletteEncoder paletteEncoder = GetOrAdd<PaletteEncoder>(runtimeObject);
            ONNXInferenceController inferenceController = GetOrAdd<ONNXInferenceController>(runtimeObject);
            HeightPostprocessor heightPostprocessor = GetOrAdd<HeightPostprocessor>(runtimeObject);
            TerrainTexturePostprocessor texturePostprocessor = GetOrAdd<TerrainTexturePostprocessor>(runtimeObject);
            TerrainBuilder terrainBuilder = GetOrAdd<TerrainBuilder>(runtimeObject);
            VegetationSpawner vegetationSpawner = GetOrAdd<VegetationSpawner>(runtimeObject);
            TerrainGenerator terrainGenerator = GetOrAdd<TerrainGenerator>(runtimeObject);
            ExampleSceneController sceneController = GetOrAdd<ExampleSceneController>(runtimeObject);

            terrainBuilder.targetTerrain = terrain;
            terrainBuilder.useDelayedLod = false;
            terrain.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;

            terrainGenerator.paletteEncoder = paletteEncoder;
            terrainGenerator.inferenceController = inferenceController;
            terrainGenerator.heightPostprocessor = heightPostprocessor;
            terrainGenerator.texturePostprocessor = texturePostprocessor;
            terrainGenerator.terrainBuilder = terrainBuilder;
            terrainGenerator.vegetationSpawner = vegetationSpawner;
            vegetationSpawner.palette = palette;
            vegetationSpawner.targetTerrain = terrain;

            sceneController.previewProfile = profile;
            sceneController.terrainGenerator = terrainGenerator;
            sceneController.defaultBiomeTexture = demoTexture;
            sceneController.applyProfileBeforeGeneration = true;
            profile.ApplyTo(terrainGenerator);

            EditorUtility.SetDirty(runtimeObject);
            EditorUtility.SetDirty(terrain);
            EditorUtility.SetDirty(terrain.terrainData);
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Selection.activeGameObject = runtimeObject;

            Debug.Log("Terrain GAN demo scene is ready. Select TerrainGANRuntime and press Generate Default Terrain in the Inspector.");
            return sceneController;
        }

        private static Terrain FindOrCreateTerrain(Terrain preferredTerrain)
        {
            if (preferredTerrain != null)
            {
                preferredTerrain.name = TerrainObjectName;
                preferredTerrain.terrainData.heightmapResolution = 257;
                preferredTerrain.terrainData.size = new Vector3(1000f, 220f, 1000f);
                return preferredTerrain;
            }

            Terrain terrain = Object.FindFirstObjectByType<Terrain>();
            if (terrain != null)
            {
                terrain.name = TerrainObjectName;
                terrain.terrainData.heightmapResolution = 257;
                terrain.terrainData.size = new Vector3(1000f, 220f, 1000f);
                return terrain;
            }

            Directory.CreateDirectory(TerrainDataFolder);
            TerrainData terrainData = AssetDatabase.LoadAssetAtPath<TerrainData>(TerrainDataPath);
            if (terrainData == null)
            {
                terrainData = new TerrainData
                {
                    heightmapResolution = 257,
                    size = new Vector3(1000f, 220f, 1000f)
                };
                AssetDatabase.CreateAsset(terrainData, TerrainDataPath);
            }

            GameObject terrainObject = Terrain.CreateTerrainGameObject(terrainData);
            terrainObject.name = TerrainObjectName;
            return terrainObject.GetComponent<Terrain>();
        }

        private static bool DefaultSceneAssetsExist()
        {
            return AssetDatabase.LoadAssetAtPath<BiomePalette>(PalettePath) != null &&
                   AssetDatabase.LoadAssetAtPath<Texture2D>(DemoTexturePath) != null &&
                   AssetDatabase.LoadAssetAtPath<TerrainGANPreviewProfile>(ProfilePath) != null;
        }

        private static T GetOrAdd<T>(GameObject gameObject) where T : Component
        {
            T component = gameObject.GetComponent<T>();
            if (component == null)
            {
                component = gameObject.AddComponent<T>();
            }

            return component;
        }
    }
}
