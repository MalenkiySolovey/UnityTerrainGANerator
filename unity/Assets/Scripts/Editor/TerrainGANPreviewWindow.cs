using Unity.InferenceEngine;
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
    public sealed class TerrainGANPreviewWindow : EditorWindow
    {
        private const string DefaultProfilePath = "Assets/TerrainGAN/Profiles/DemoPreviewProfile.asset";

        private TerrainGANPreviewProfile profile;
        private ExampleSceneController controller;
        private Terrain targetTerrain;
        private Vector2 scroll;
        private bool showInputs = true;
        private bool showInference = true;
        private bool showOrientation = true;
        private bool showTerrain = true;
        private bool showObjectPlacement = true;
        private bool showPostprocess = true;
        private bool showExport = true;

        [MenuItem("Tools/Terrain GAN/Preview Window")]
        public static void Open()
        {
            TerrainGANPreviewWindow window = GetWindow<TerrainGANPreviewWindow>("Terrain GAN Preview");
            window.minSize = new Vector2(440f, 620f);
            window.LoadDefaults();
        }

        private void OnEnable()
        {
            LoadDefaults();
        }

        private void OnDisable()
        {
            DisposePreviewInference();
        }

        private void OnGUI()
        {
            scroll = EditorGUILayout.BeginScrollView(scroll);
            EditorGUILayout.LabelField("Terrain GAN Preview Tool", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            DrawProfileBar();

            using (new EditorGUI.DisabledScope(profile == null))
            {
                DrawInputSettings();
                DrawInferenceSettings();
                DrawOrientationSettings();
                DrawTerrainSettings();
                DrawObjectPlacementSettings();
                DrawPostprocessSettings();
            }

            EditorGUILayout.Space();
            DrawActionButtons();
            DrawExportButtons();
            DrawStatus();
            DrawTexturePreview();
            EditorGUILayout.EndScrollView();
        }

        private void DrawProfileBar()
        {
            EditorGUI.BeginChangeCheck();
            profile = (TerrainGANPreviewProfile)EditorGUILayout.ObjectField("Preview Profile", profile, typeof(TerrainGANPreviewProfile), false);
            if (EditorGUI.EndChangeCheck())
            {
                MarkProfileDirty();
            }

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Use Demo Defaults"))
            {
                TerrainGANDemoAssetCreator.CreateDemoAssets();
                profile = AssetDatabase.LoadAssetAtPath<TerrainGANPreviewProfile>(DefaultProfilePath);
                controller = Object.FindFirstObjectByType<ExampleSceneController>();
                GUI.FocusControl(null);
            }

            if (GUILayout.Button("Select Profile") && profile != null)
            {
                Selection.activeObject = profile;
                EditorGUIUtility.PingObject(profile);
            }

            EditorGUILayout.EndHorizontal();

            controller = (ExampleSceneController)EditorGUILayout.ObjectField("Scene Controller", controller, typeof(ExampleSceneController), true);
            targetTerrain = (Terrain)EditorGUILayout.ObjectField("Target Terrain", targetTerrain, typeof(Terrain), true);
            if (GUILayout.Button("Use Scene Terrain"))
            {
                targetTerrain = Object.FindFirstObjectByType<Terrain>();
                AssignTargetTerrain();
            }
        }

        private void DrawInputSettings()
        {
            showInputs = EditorGUILayout.Foldout(showInputs, "Input Data", true);
            if (!showInputs)
            {
                return;
            }

            EditorGUI.indentLevel++;
            EditorGUI.BeginChangeCheck();
            profile.biomeTexture = (Texture2D)EditorGUILayout.ObjectField("Biome Texture", profile.biomeTexture, typeof(Texture2D), false);
            profile.modelAsset = (ModelAsset)EditorGUILayout.ObjectField("ONNX Model", profile.modelAsset, typeof(ModelAsset), false);
            profile.palette = (BiomePalette)EditorGUILayout.ObjectField("Biome Palette", profile.palette, typeof(BiomePalette), false);
            if (EditorGUI.EndChangeCheck())
            {
                MarkProfileDirty();
            }

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Make Texture Readable"))
            {
                TerrainGANTextureImportUtility.EnsureReadable(profile.biomeTexture, true);
            }

            if (GUILayout.Button("Select Texture") && profile.biomeTexture != null)
            {
                Selection.activeObject = profile.biomeTexture;
                EditorGUIUtility.PingObject(profile.biomeTexture);
            }

            EditorGUILayout.EndHorizontal();
            EditorGUI.indentLevel--;
        }

        private void DrawOrientationSettings()
        {
            showOrientation = EditorGUILayout.Foldout(showOrientation, "Orientation", true);
            if (!showOrientation)
            {
                return;
            }

            EditorGUI.indentLevel++;
            EditorGUILayout.HelpBox("Base orientation is fixed automatically. Generated ONNX Texture and Painted Biome Layers each use their own correct alignment with the heightmap. Use these options only to mirror the final terrain.", MessageType.Info);

            EditorGUI.BeginChangeCheck();
            profile.invertOutputWidth = EditorGUILayout.Toggle("Mirror Width (X)", profile.invertOutputWidth);
            profile.invertOutputLength = EditorGUILayout.Toggle("Mirror Length (Z)", profile.invertOutputLength);
            if (EditorGUI.EndChangeCheck())
            {
                MarkProfileDirty();
            }

            if (GUILayout.Button("Reset Mirrors"))
            {
                profile.invertOutputWidth = false;
                profile.invertOutputLength = false;
                MarkProfileDirty();
            }

            EditorGUI.indentLevel--;
        }

        private void DrawInferenceSettings()
        {
            showInference = EditorGUILayout.Foldout(showInference, "ONNX Inference", true);
            if (!showInference)
            {
                return;
            }

            EditorGUI.indentLevel++;
            EditorGUI.BeginChangeCheck();
            profile.backendType = (BackendType)EditorGUILayout.EnumPopup("Backend", profile.backendType);
            profile.inputResolution = Mathf.Max(1, EditorGUILayout.IntField("Input Resolution", profile.inputResolution));
            profile.macroClassCount = Mathf.Max(1, EditorGUILayout.IntField("Macro Class Count", profile.macroClassCount));
            profile.modelInputChannels = Mathf.Max(1, EditorGUILayout.IntField("Model Input Channels", profile.modelInputChannels));
            profile.biomeInputName = EditorGUILayout.TextField("Biome Input Name", profile.biomeInputName);
            profile.terrainOutputName = EditorGUILayout.TextField("Terrain Output Name", profile.terrainOutputName);
            if (profile.modelInputChannels == profile.macroClassCount + PaletteEncoder.V124ReliefControlChannelCount)
            {
                EditorGUILayout.HelpBox("v12.4 mode: Unity sends 7 macro channels plus height_prior, detail_budget, boundary_distance, and semantic_complexity.", MessageType.Info);
            }
            else if (profile.modelInputChannels != profile.macroClassCount)
            {
                EditorGUILayout.HelpBox("Unsupported input channel count for the current encoder. Use 11 for v12.4 terrain4.", MessageType.Warning);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Terrain Variation", EditorStyles.boldLabel);
            profile.useNoiseInputs = EditorGUILayout.Toggle("Enable Variation", profile.useNoiseInputs);
            EditorGUILayout.HelpBox(
                profile.useNoiseInputs
                    ? "Enabled: the same biome image can produce different terrain details using Seed and Strength."
                    : "Disabled: deterministic mode. The same biome image gives the same terrain every time.",
                MessageType.Info
            );

            using (new EditorGUI.DisabledScope(!profile.useNoiseInputs))
            {
                profile.noiseSeed = Mathf.Max(0, EditorGUILayout.IntField("Variation Seed", profile.noiseSeed));
                profile.noiseStrength = EditorGUILayout.Slider("Variation Strength", profile.noiseStrength, 0f, 1f);
                profile.noiseDistribution = (LatentNoiseDistribution)EditorGUILayout.EnumPopup("Variation Distribution", profile.noiseDistribution);
                profile.useTrainingNoiseInputs = EditorGUILayout.Toggle("Use Training Preview Noise", profile.useTrainingNoiseInputs);
                profile.trainingNoiseStrength = EditorGUILayout.Slider("Training Noise Strength", profile.trainingNoiseStrength, 0f, 1f);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Advanced ONNX Inputs", EditorStyles.boldLabel);
            profile.noiseInputName = EditorGUILayout.TextField("Noise Vector Input Name", profile.noiseInputName);
            profile.noiseDim = Mathf.Max(1, EditorGUILayout.IntField("Noise Vector Size", profile.noiseDim));
            profile.trainingNoiseInputPrefix = EditorGUILayout.TextField("Training Noise Input Prefix", profile.trainingNoiseInputPrefix);
            if (EditorGUI.EndChangeCheck())
            {
                MarkProfileDirty();
            }

            if (GUILayout.Button("New Random Variant"))
            {
                profile.useNoiseInputs = true;
                profile.noiseSeed = UnityEngine.Random.Range(1, int.MaxValue);
                MarkProfileDirty();
            }

            if (GUILayout.Button("Use v12.4 Terrain4 ONNX Contract"))
            {
                profile.biomeInputName = "biome_onehot";
                profile.noiseInputName = "noise";
                profile.terrainOutputName = "terrain";
                profile.modelInputChannels = profile.macroClassCount + PaletteEncoder.V124ReliefControlChannelCount;
                profile.textureMode = TerrainTextureMode.GeneratedTexture;
                profile.generatedTextureRedChannel = 0;
                profile.generatedTextureGreenChannel = 1;
                profile.generatedTextureBlueChannel = 2;
                profile.generatedHeightChannel = 3;
                profile.invertOutputWidth = false;
                profile.invertOutputLength = false;
                profile.useNoiseInputs = true;
                profile.noiseDistribution = LatentNoiseDistribution.Uniform01;
                profile.noiseStrength = 1f;
                profile.useTrainingNoiseInputs = true;
                profile.trainingNoiseInputPrefix = "train_noise_";
                profile.trainingNoiseStrength = 1f;
                ModelAsset v124Model = LoadFirstModelAsset(
                    "Assets/Models/spade_terrain_epoch116.onnx",
                    "Assets/Models/spade_terrain_epoch128.onnx",
                    "Assets/Models/spade_terrain_epoch90.onnx",
                    "Assets/Models/spade_terrain_rgb_height_v12_4_feature_supervised_256.onnx"
                );
                if (v124Model != null)
                {
                    profile.modelAsset = v124Model;
                }
                MarkProfileDirty();
            }

            EditorGUI.indentLevel--;
        }

        private void DrawTerrainSettings()
        {
            showTerrain = EditorGUILayout.Foldout(showTerrain, "Unity Terrain", true);
            if (!showTerrain)
            {
                return;
            }

            EditorGUI.indentLevel++;
            EditorGUI.BeginChangeCheck();
            Terrain selectedTerrain = (Terrain)EditorGUILayout.ObjectField("Target Terrain", targetTerrain, typeof(Terrain), true);
            if (selectedTerrain != targetTerrain)
            {
                targetTerrain = selectedTerrain;
                AssignTargetTerrain();
            }

            profile.terrainSize = EditorGUILayout.Vector3Field("Terrain Size", profile.terrainSize);
            profile.heightmapResolution = Mathf.Max(2, EditorGUILayout.IntField("Heightmap Resolution", profile.heightmapResolution));
            profile.alphamapResolution = Mathf.Max(16, EditorGUILayout.IntField("Alphamap Resolution", profile.alphamapResolution));
            profile.useDelayedLod = EditorGUILayout.Toggle("Use Delayed LOD", profile.useDelayedLod);

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Model Detail"))
            {
                SetTerrainQuality(257, 256, false, 1);
            }

            if (GUILayout.Button("High Detail"))
            {
                SetTerrainQuality(513, 512, false, 1);
            }

            if (GUILayout.Button("Ultra Detail"))
            {
                SetTerrainQuality(1025, 1024, false, 1);
            }

            EditorGUILayout.EndHorizontal();
            EditorGUILayout.HelpBox("Model inference stays 256x256. Higher terrain detail upsamples height geometry and uses denser biome alphamaps for painted layers.", MessageType.Info);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Tiled ONNX Generation", EditorStyles.boldLabel);
            EditorGUI.BeginChangeCheck();
            profile.useTiledGeneration = EditorGUILayout.Toggle("Enable Tiled Generation", profile.useTiledGeneration);
            using (new EditorGUI.DisabledScope(!profile.useTiledGeneration))
            {
                profile.tileGrid = Mathf.Clamp(EditorGUILayout.IntField("Tile Grid", profile.tileGrid), 1, 8);
                profile.tileOverlapPixels = Mathf.Clamp(EditorGUILayout.IntField("Tile Overlap Pixels", profile.tileOverlapPixels), 0, Mathf.Max(0, profile.inputResolution / 2 - 1));
                profile.usePerTileVariation = EditorGUILayout.Toggle("Per-Tile Variation Seeds", profile.usePerTileVariation);
            }

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Single 1x1"))
            {
                SetTerrainQuality(257, 256, false, 1);
            }

            if (GUILayout.Button("Tiled 2x2"))
            {
                SetTerrainQuality(513, 512, true, 2);
            }

            if (GUILayout.Button("Tiled 4x4"))
            {
                SetTerrainQuality(1025, 1024, true, 4);
            }

            EditorGUILayout.EndHorizontal();
            EditorGUILayout.HelpBox(
                profile.useTiledGeneration
                    ? "Tiled mode runs ONNX once per tile, blends overlaps, then applies height remap and texture postprocess to the assembled terrain. 4x4 gives more real generated pixels but is slower."
                    : "Single mode runs ONNX once and only upsamples the result when Terrain resolution is higher.",
                MessageType.Info
            );
            if (EditorGUI.EndChangeCheck())
            {
                MarkProfileDirty();
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Texture Source", EditorStyles.boldLabel);
            string[] textureLabels = { "None", "Painted Biome Layers", "Generated ONNX Texture" };
            profile.textureMode = (TerrainTextureMode)GUILayout.Toolbar((int)profile.textureMode, textureLabels);
            profile.paintTerrainLayers = profile.textureMode == TerrainTextureMode.PaintedBiomeMap;
            using (new EditorGUI.DisabledScope(profile.textureMode != TerrainTextureMode.GeneratedTexture))
            {
                profile.generatedTextureRedChannel = Mathf.Max(0, EditorGUILayout.IntField("Generated Red Channel", profile.generatedTextureRedChannel));
                profile.generatedTextureGreenChannel = Mathf.Max(0, EditorGUILayout.IntField("Generated Green Channel", profile.generatedTextureGreenChannel));
                profile.generatedTextureBlueChannel = Mathf.Max(0, EditorGUILayout.IntField("Generated Blue Channel", profile.generatedTextureBlueChannel));
                EditorGUILayout.Space();
                EditorGUILayout.LabelField("Generated Texture Appearance", EditorStyles.boldLabel);
                EditorGUILayout.HelpBox("These controls are applied after ONNX inference. They do not change the model input or neural network output.", MessageType.Info);
                profile.textureSaturation = EditorGUILayout.Slider("Saturation", profile.textureSaturation, 0f, 2f);
                profile.textureBrightness = EditorGUILayout.Slider("Brightness", profile.textureBrightness, 0f, 2f);
                profile.textureContrast = EditorGUILayout.Slider("Contrast", profile.textureContrast, 0f, 2f);
                profile.textureGamma = EditorGUILayout.Slider("Gamma", profile.textureGamma, 0.25f, 3f);
                profile.textureTint = EditorGUILayout.ColorField("Tint", profile.textureTint);
                if (GUILayout.Button("Reset Texture Appearance"))
                {
                    profile.textureSaturation = 1f;
                    profile.textureBrightness = 1f;
                    profile.textureContrast = 1f;
                    profile.textureGamma = 1f;
                    profile.textureTint = Color.white;
                }
            }

            profile.generatedHeightChannel = Mathf.Max(0, EditorGUILayout.IntField("Generated Height Channel", profile.generatedHeightChannel));
            if (EditorGUI.EndChangeCheck())
            {
                MarkProfileDirty();
            }

            EditorGUI.indentLevel--;
        }

        private void DrawObjectPlacementSettings()
        {
            EditorGUILayout.Space();
            showObjectPlacement = EditorGUILayout.Foldout(showObjectPlacement, "Object Placement", true);
            if (!showObjectPlacement)
            {
                return;
            }

            EditorGUI.indentLevel++;
            EditorGUI.BeginChangeCheck();
            profile.enableObjectPlacement = EditorGUILayout.Toggle("Place Objects On Terrain", profile.enableObjectPlacement);
            profile.clearObjectsWhenDisabled = EditorGUILayout.Toggle("Clear When Disabled", profile.clearObjectsWhenDisabled);
            using (new EditorGUI.DisabledScope(!profile.enableObjectPlacement))
            {
                profile.useObjectRulesPlacement = EditorGUILayout.Toggle("Use Macro/Microbiome Rules", profile.useObjectRulesPlacement);
                profile.useLegacyShortcutPlacement = EditorGUILayout.Toggle("Use Legacy Shortcuts", profile.useLegacyShortcutPlacement);
                profile.objectRulePlacementResolution = Mathf.Clamp(EditorGUILayout.IntField("Rule Scan Resolution", profile.objectRulePlacementResolution), 8, 512);
                profile.objectRuleDensityScale = Mathf.Max(0f, EditorGUILayout.FloatField("Rule Density Scale", profile.objectRuleDensityScale));
                profile.maxObjectRuleInstances = Mathf.Max(0, EditorGUILayout.IntField("Max Rule Objects", profile.maxObjectRuleInstances));
                profile.objectPlacementResolution = Mathf.Clamp(EditorGUILayout.IntField("Legacy Scan Resolution", profile.objectPlacementResolution), 8, 512);
            }

            EditorGUILayout.HelpBox(
                profile.enableObjectPlacement
                    ? "Objects are spawned after terrain generation. Use the Biome Palette Manager to assign prefabs to macro/microbiome rules."
                    : "Object placement is disabled. If Clear When Disabled is enabled, Apply Settings and Generate Terrain remove previously spawned objects.",
                MessageType.Info);

            using (new EditorGUI.DisabledScope(controller == null || controller.terrainGenerator == null))
            {
                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Respawn Objects"))
                {
                    ApplyProfile();
                    controller.terrainGenerator.RespawnVegetation();
                    MarkTerrainDirty();
                }

                if (GUILayout.Button("Clear Objects"))
                {
                    if (controller.terrainGenerator.vegetationSpawner != null)
                    {
                        controller.terrainGenerator.vegetationSpawner.ClearSpawnedObjects();
                    }
                    MarkTerrainDirty();
                }
                EditorGUILayout.EndHorizontal();
            }

            if (EditorGUI.EndChangeCheck())
            {
                MarkProfileDirty();
            }
            EditorGUI.indentLevel--;
        }

        private void DrawPostprocessSettings()
        {
            showPostprocess = EditorGUILayout.Foldout(showPostprocess, "Height Postprocess", true);
            if (!showPostprocess)
            {
                return;
            }

            EditorGUI.indentLevel++;
            EditorGUI.BeginChangeCheck();
            profile.clamp01 = EditorGUILayout.Toggle("Clamp 0..1", profile.clamp01);
            profile.resize256To257 = EditorGUILayout.Toggle("Resize Height To Terrain Resolution", profile.resize256To257);
            profile.remapWithPercentiles = EditorGUILayout.Toggle("Remap With Percentiles", profile.remapWithPercentiles);
            profile.lowerPercentile = EditorGUILayout.Slider("Lower Percentile", profile.lowerPercentile, 0f, 0.25f);
            profile.upperPercentile = EditorGUILayout.Slider("Upper Percentile", profile.upperPercentile, 0.75f, 1f);
            profile.outputMin = EditorGUILayout.Slider("Output Min", profile.outputMin, 0f, 1f);
            profile.outputMax = EditorGUILayout.Slider("Output Max", profile.outputMax, 0f, 1f);
            profile.smoothingIterations = Mathf.Max(0, EditorGUILayout.IntField("Smoothing Iterations", profile.smoothingIterations));
            profile.smoothingStrength = EditorGUILayout.Slider("Smoothing Strength", profile.smoothingStrength, 0f, 1f);
            if (EditorGUI.EndChangeCheck())
            {
                MarkProfileDirty();
            }

            EditorGUI.indentLevel--;
        }

        private void DrawActionButtons()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Actions", EditorStyles.boldLabel);

            if (GUILayout.Button("Setup Scene"))
            {
                TerrainGANPreviewProfile selectedProfile = profile;
                controller = TerrainGANSceneSetup.SetupDemoSceneAndGetController(targetTerrain, selectedProfile);

                if (controller != null && profile != null)
                {
                    controller.previewProfile = profile;
                    AssignTargetTerrain();
                    ApplyProfile();
                }
            }

            using (new EditorGUI.DisabledScope(controller == null || profile == null))
            {
                if (GUILayout.Button("Apply Settings To Scene"))
                {
                    ApplyProfile();
                    if (profile != null &&
                        !profile.enableObjectPlacement &&
                        profile.clearObjectsWhenDisabled &&
                        controller != null &&
                        controller.terrainGenerator != null &&
                        controller.terrainGenerator.vegetationSpawner != null)
                    {
                        controller.terrainGenerator.vegetationSpawner.ClearSpawnedObjects();
                    }
                }

                if (GUILayout.Button("Generate Terrain"))
                {
                    GeneratePreview();
                }
            }
        }

        private void DrawExportButtons()
        {
            showExport = EditorGUILayout.Foldout(showExport, "Export", true);
            if (!showExport)
            {
                return;
            }

            using (new EditorGUI.DisabledScope(controller == null || controller.terrainGenerator == null))
            {
                if (GUILayout.Button("Export Heightmap PNG"))
                {
                    PingAsset(TerrainGANPreviewExportUtility.ExportHeightmapPng(GetTargetTerrain()));
                }

                if (GUILayout.Button("Export Generated Texture PNG"))
                {
                    PingAsset(TerrainGANPreviewExportUtility.ExportGeneratedTexture(controller.terrainGenerator.LastGeneratedTexture));
                }

                if (GUILayout.Button("Export Painted Biome Map PNG"))
                {
                    Texture2D texture = profile != null && profile.biomeTexture != null
                        ? profile.biomeTexture
                        : controller.defaultBiomeTexture;
                    PingAsset(TerrainGANPreviewExportUtility.ExportBiomeTexture(texture));
                }

                if (GUILayout.Button("Export Model Macro Input PNG"))
                {
                    Texture2D texture = profile != null && profile.biomeTexture != null
                        ? profile.biomeTexture
                        : controller.defaultBiomeTexture;
                    BiomePalette palette = profile != null ? profile.palette : null;
                    PingAsset(TerrainGANPreviewExportUtility.ExportMacroBiomeTexture(texture, palette));
                }
            }
        }

        private void DrawStatus()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);

            if (profile == null)
            {
                EditorGUILayout.HelpBox("No Preview Profile selected. Press Use Demo Defaults.", MessageType.Warning);
                return;
            }

            if (profile.biomeTexture == null)
            {
                EditorGUILayout.HelpBox("Biome Texture is not assigned.", MessageType.Warning);
            }
            else if (!profile.biomeTexture.isReadable)
            {
                EditorGUILayout.HelpBox("Biome Texture is not readable. Press Make Texture Readable or Generate Terrain to fix imported assets automatically.", MessageType.Info);
            }

            if (profile.modelAsset == null)
            {
                EditorGUILayout.HelpBox("ONNX Model is not assigned.", MessageType.Warning);
            }

            if (profile.palette == null)
            {
                EditorGUILayout.HelpBox("Biome Palette is not assigned.", MessageType.Warning);
            }

            if (profile.textureMode == TerrainTextureMode.GeneratedTexture)
            {
                EditorGUILayout.HelpBox("Generated Texture mode uses ONNX output channels 0..2 as the Terrain diffuse texture and channel 3 as height.", MessageType.Info);
            }

            if (controller == null)
            {
                EditorGUILayout.HelpBox("Scene Controller is missing. Press Setup Scene.", MessageType.Info);
            }
        }

        private void DrawTexturePreview()
        {
            if (controller == null || controller.terrainGenerator == null)
            {
                return;
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Texture Preview", EditorStyles.boldLabel);
            EditorGUILayout.BeginHorizontal();
            DrawPreviewTexture("Biome Map", profile != null ? profile.biomeTexture : null);
            DrawPreviewTexture("Applied Texture", controller.terrainGenerator.LastAppliedTexture);
            EditorGUILayout.EndHorizontal();
        }

        private static void DrawPreviewTexture(string label, Texture texture)
        {
            EditorGUILayout.BeginVertical(GUILayout.MinWidth(180f));
            EditorGUILayout.LabelField(label, EditorStyles.miniBoldLabel);
            Rect rect = GUILayoutUtility.GetRect(180f, 180f, GUILayout.ExpandWidth(true));
            if (texture != null)
            {
                EditorGUI.DrawPreviewTexture(rect, texture, null, ScaleMode.ScaleToFit);
            }
            else
            {
                EditorGUI.DrawRect(rect, new Color(0.16f, 0.16f, 0.16f));
                GUI.Label(rect, "No texture", EditorStyles.centeredGreyMiniLabel);
            }

            EditorGUILayout.EndVertical();
        }

        private void LoadDefaults()
        {
            profile = AssetDatabase.LoadAssetAtPath<TerrainGANPreviewProfile>(DefaultProfilePath);
            controller = Object.FindFirstObjectByType<ExampleSceneController>();
            targetTerrain = GetTargetTerrain();
            if (targetTerrain == null)
            {
                targetTerrain = Object.FindFirstObjectByType<Terrain>();
            }
        }

        private void ApplyProfile()
        {
            if (controller == null || profile == null)
            {
                return;
            }

            controller.previewProfile = profile;
            AssignTargetTerrain();
            controller.ApplyPreviewProfile();
            EditorUtility.SetDirty(controller);

            if (controller.terrainGenerator != null)
            {
                EditorUtility.SetDirty(controller.terrainGenerator);
            }

            Terrain terrain = GetTargetTerrain();
            if (terrain != null)
            {
                EditorUtility.SetDirty(terrain);
                if (terrain.terrainData != null)
                {
                    EditorUtility.SetDirty(terrain.terrainData);
                }
            }

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }

        private void SetTerrainQuality(int heightResolution, int alphaResolution, bool useTiledGeneration, int tileGrid)
        {
            profile.heightmapResolution = heightResolution;
            profile.alphamapResolution = alphaResolution;
            profile.resize256To257 = true;
            profile.useTiledGeneration = useTiledGeneration;
            profile.tileGrid = Mathf.Max(1, tileGrid);
            if (useTiledGeneration && profile.tileOverlapPixels <= 0)
            {
                profile.tileOverlapPixels = 32;
            }
            MarkProfileDirty();
        }

        private void GeneratePreview()
        {
            if (controller == null || profile == null)
            {
                return;
            }

            TerrainGANTextureImportUtility.EnsureReadable(profile.biomeTexture, true);
            ApplyProfile();
            controller.GenerateDefault();

            Terrain terrain = GetTargetTerrain();
            if (terrain != null && terrain.terrainData != null)
            {
                EditorUtility.SetDirty(terrain.terrainData);
            }

            MarkTerrainDirty();

            SceneView.RepaintAll();
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }

        private void MarkTerrainDirty()
        {
            Terrain terrain = GetTargetTerrain();
            if (terrain != null)
            {
                EditorUtility.SetDirty(terrain);
                if (terrain.terrainData != null)
                {
                    EditorUtility.SetDirty(terrain.terrainData);
                }
            }

            if (controller != null && controller.terrainGenerator != null && controller.terrainGenerator.vegetationSpawner != null)
            {
                EditorUtility.SetDirty(controller.terrainGenerator.vegetationSpawner);
            }

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }

        private Terrain GetTargetTerrain()
        {
            if (controller == null ||
                controller.terrainGenerator == null ||
                controller.terrainGenerator.terrainBuilder == null)
            {
                return null;
            }

            return controller.terrainGenerator.terrainBuilder.targetTerrain;
        }

        private void AssignTargetTerrain()
        {
            if (controller == null ||
                controller.terrainGenerator == null ||
                controller.terrainGenerator.terrainBuilder == null ||
                targetTerrain == null)
            {
                return;
            }

            controller.terrainGenerator.terrainBuilder.targetTerrain = targetTerrain;
            EditorUtility.SetDirty(controller.terrainGenerator.terrainBuilder);
        }

        private void MarkProfileDirty()
        {
            if (profile != null)
            {
                EditorUtility.SetDirty(profile);
            }
        }

        private static void PingAsset(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                return;
            }

            Object asset = AssetDatabase.LoadAssetAtPath<Object>(path);
            if (asset != null)
            {
                Selection.activeObject = asset;
                EditorGUIUtility.PingObject(asset);
            }
        }

        private static ModelAsset LoadFirstModelAsset(params string[] paths)
        {
            foreach (string path in paths)
            {
                ModelAsset model = AssetDatabase.LoadAssetAtPath<ModelAsset>(path);
                if (model != null)
                {
                    return model;
                }
            }

            return null;
        }

        private void DisposePreviewInference()
        {
            if (controller == null ||
                controller.terrainGenerator == null ||
                controller.terrainGenerator.inferenceController == null)
            {
                return;
            }

            controller.terrainGenerator.inferenceController.Dispose();
        }
    }
}
