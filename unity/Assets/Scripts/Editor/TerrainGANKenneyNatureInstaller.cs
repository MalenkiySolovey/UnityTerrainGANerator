#if UNITY_EDITOR
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Biomes;
using UnityTerrainGANerator.Preview;

namespace UnityTerrainGANerator.Editor
{
    public static class TerrainGANKenneyNatureInstaller
    {
        private const string KenneyRoot = "Assets/TerrainGAN/ThirdParty/KenneyNatureKit";
        private const string ModelFolder = KenneyRoot + "/Models/FBX";
        private const string PrefabFolder = KenneyRoot + "/Prefabs";
        private const string PalettePath = "Assets/TerrainGAN/Palettes/Dataset_BiomePalette.asset";
        private const string ProfilePath = "Assets/TerrainGAN/Profiles/DemoPreviewProfile.asset";

        private static readonly string[] PrefabNames =
        {
            "tree_default",
            "tree_detailed",
            "tree_oak",
            "tree_small",
            "tree_pineDefaultA",
            "tree_pineDefaultB",
            "tree_pineTallA",
            "tree_pineTallB",
            "tree_palmTall",
            "tree_palmShort",
            "tree_cone",
            "plant_bush",
            "plant_bushDetailed",
            "plant_bushSmall",
            "plant_flatShort",
            "plant_flatTall",
            "grass",
            "grass_large",
            "grass_leafs",
            "flower_yellowA",
            "flower_yellowB",
            "flower_redA",
            "flower_redB",
            "flower_purpleA",
            "flower_purpleB",
            "mushroom_red",
            "mushroom_tan",
            "lily_large",
            "lily_small",
            "cactus_short",
            "cactus_tall",
            "rock_largeA",
            "rock_largeB",
            "rock_largeC",
            "rock_smallA",
            "rock_smallB",
            "rock_smallC",
            "stone_largeA",
            "stone_largeB",
            "stone_smallA",
            "stone_smallB",
            "stone_tallA",
            "stone_tallB",
            "log",
            "log_large",
            "stump_round",
            "stump_old"
        };

        [MenuItem("Tools/Terrain GAN/Assets/Install Kenney Nature Objects")]
        public static void InstallAndAssign()
        {
            if (!AssetDatabase.IsValidFolder(KenneyRoot) || !AssetDatabase.IsValidFolder(ModelFolder))
            {
                Debug.LogError("Kenney Nature Kit is not installed. Expected FBX files at " + ModelFolder);
                return;
            }

            EnsureFolder(PrefabFolder);
            AssetDatabase.ImportAsset(KenneyRoot, ImportAssetOptions.ImportRecursive | ImportAssetOptions.ForceUpdate);
            Dictionary<string, GameObject> prefabs = CreateSelectedPrefabs();
            AssignPalette(prefabs);
            ConfigureDefaultProfile();
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log($"Terrain GAN Kenney Nature Kit ready: {prefabs.Count} prefabs created/updated and assigned to biome placement rules.");
        }

        private static Dictionary<string, GameObject> CreateSelectedPrefabs()
        {
            Dictionary<string, GameObject> prefabs = new Dictionary<string, GameObject>();
            foreach (string prefabName in PrefabNames)
            {
                string modelPath = $"{ModelFolder}/{prefabName}.fbx";
                GameObject model = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
                if (model == null)
                {
                    Debug.LogWarning($"Kenney model missing: {modelPath}");
                    continue;
                }

                string prefabPath = $"{PrefabFolder}/{prefabName}.prefab";
                GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(model);
                if (instance == null)
                {
                    Debug.LogWarning($"Could not instantiate model for prefab: {modelPath}");
                    continue;
                }

                instance.name = prefabName;
                PrefabUtility.SaveAsPrefabAsset(instance, prefabPath);
                Object.DestroyImmediate(instance);
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
                if (prefab != null)
                {
                    prefabs[prefabName] = prefab;
                }
            }

            return prefabs;
        }

        private static void AssignPalette(Dictionary<string, GameObject> prefabs)
        {
            BiomePalette palette = AssetDatabase.LoadAssetAtPath<BiomePalette>(PalettePath);
            if (palette == null)
            {
                Debug.LogError("Biome palette not found: " + PalettePath);
                return;
            }

            Undo.RecordObject(palette, "Assign Kenney macrobiome objects");
            palette.EnsureMacroBiomes();
            SetMacroRules(palette, 0, 0.55f,
                Rule("Water Lilies", "lily_small", BiomeObjectCategory.Detail, 18f, 0f, 0.28f, 0f, 8f, 0.8f, 1.3f, 5f),
                Rule("Large Water Lilies", "lily_large", BiomeObjectCategory.Detail, 8f, 0f, 0.25f, 0f, 7f, 0.8f, 1.25f, 7f));
            SetMacroRules(palette, 1, 0.85f,
                Rule("Coastal Palms", "tree_palmShort", BiomeObjectCategory.Tree, 6f, 0.02f, 0.42f, 0f, 18f, 0.85f, 1.25f, 18f),
                Rule("Coastal Rocks", "rock_smallA", BiomeObjectCategory.Rock, 18f, 0.02f, 0.48f, 0f, 24f, 0.7f, 1.4f, 7f),
                Rule("Lowland Bushes", "plant_bushSmall", BiomeObjectCategory.Detail, 35f, 0.04f, 0.55f, 0f, 22f, 0.75f, 1.25f, 5f));
            SetMacroRules(palette, 2, 1f,
                Rule("Plain Grass", "grass", BiomeObjectCategory.Detail, 95f, 0.05f, 0.65f, 0f, 22f, 0.7f, 1.15f, 1.3f),
                Rule("Plain Flowers", "flower_yellowA", BiomeObjectCategory.Detail, 28f, 0.05f, 0.65f, 0f, 18f, 0.8f, 1.25f, 2.2f),
                Rule("Plain Bushes", "plant_bush", BiomeObjectCategory.Detail, 18f, 0.06f, 0.68f, 0f, 24f, 0.8f, 1.25f, 5f));
            SetMacroRules(palette, 3, 0.9f,
                Rule("Hill Trees", "tree_default", BiomeObjectCategory.Tree, 10f, 0.12f, 0.78f, 0f, 32f, 0.85f, 1.25f, 16f),
                Rule("Hill Rocks", "rock_smallB", BiomeObjectCategory.Rock, 22f, 0.12f, 0.82f, 4f, 38f, 0.75f, 1.45f, 7f),
                Rule("Hill Bushes", "plant_bushDetailed", BiomeObjectCategory.Detail, 24f, 0.1f, 0.72f, 0f, 28f, 0.8f, 1.2f, 4.5f));
            SetMacroRules(palette, 4, 0.85f,
                Rule("Highland Pines", "tree_pineDefaultA", BiomeObjectCategory.Tree, 9f, 0.18f, 0.88f, 0f, 34f, 0.8f, 1.25f, 18f),
                Rule("Highland Stones", "stone_smallA", BiomeObjectCategory.Rock, 30f, 0.18f, 0.9f, 4f, 45f, 0.75f, 1.5f, 7f),
                Rule("Highland Grass Tufts", "grass_leafs", BiomeObjectCategory.Detail, 35f, 0.16f, 0.82f, 0f, 30f, 0.8f, 1.25f, 2.5f));
            SetMacroRules(palette, 5, 0.9f,
                Rule("Mountain Rocks", "rock_largeA", BiomeObjectCategory.Rock, 28f, 0.25f, 0.96f, 10f, 72f, 0.65f, 1.6f, 8f),
                Rule("Mountain Stones", "stone_tallA", BiomeObjectCategory.Rock, 12f, 0.28f, 0.98f, 12f, 76f, 0.65f, 1.4f, 10f),
                Rule("Sparse Mountain Pines", "tree_pineTallA", BiomeObjectCategory.Tree, 3f, 0.25f, 0.78f, 0f, 28f, 0.75f, 1.15f, 24f));
            SetMacroRules(palette, 6, 0.65f,
                Rule("Peak Stones", "stone_tallB", BiomeObjectCategory.Rock, 13f, 0.45f, 1f, 14f, 80f, 0.6f, 1.35f, 10f),
                Rule("Peak Rocks", "rock_largeC", BiomeObjectCategory.Rock, 17f, 0.45f, 1f, 12f, 82f, 0.6f, 1.45f, 9f));

            EditorUtility.SetDirty(palette);

            if (palette.biomes != null)
            {
                foreach (BiomeDefinition biome in palette.biomes)
                {
                    if (biome == null)
                    {
                        continue;
                    }

                    Undo.RecordObject(biome, "Assign Kenney microbiome objects");
                    AssignMicroBiome(biome, prefabs);
                    EditorUtility.SetDirty(biome);
                }
            }

            void SetMacroRules(BiomePalette targetPalette, int macroIndex, float densityMultiplier, params RuleSpec[] specs)
            {
                MacroBiomeSettings macro = targetPalette.GetMacroBiomeSettings(macroIndex);
                if (macro == null)
                {
                    return;
                }

                macro.placement.densityMultiplier = densityMultiplier;
                macro.placement.objectRules = BuildRules(prefabs, specs);
            }
        }

        private static void AssignMicroBiome(BiomeDefinition biome, Dictionary<string, GameObject> prefabs)
        {
            string key = Normalize(biome.biomeName);
            biome.inheritMacroPlacement = key != "deepwater";

            switch (key)
            {
                case "deepwater":
                    biome.treePrefabs = new GameObject[0];
                    biome.detailPrefabs = new GameObject[0];
                    biome.rockPrefabs = new GameObject[0];
                    SetMicroRules(biome, prefabs, 0f);
                    break;
                case "shallowwater":
                    biome.detailPrefabs = Pick(prefabs, "lily_small", "lily_large");
                    biome.rockPrefabs = Pick(prefabs, "rock_smallA");
                    SetMicroRules(biome, prefabs, 0.7f,
                        Rule("Shallow Lilies", "lily_small", BiomeObjectCategory.Detail, 28f, 0f, 0.25f, 0f, 7f, 0.8f, 1.25f, 4.5f),
                        Rule("Shallow Rocks", "rock_smallA", BiomeObjectCategory.Rock, 8f, 0f, 0.28f, 0f, 12f, 0.7f, 1.25f, 7f));
                    break;
                case "forest":
                    biome.treePrefabs = Pick(prefabs, "tree_oak", "tree_detailed", "tree_pineTallA");
                    biome.detailPrefabs = Pick(prefabs, "plant_bush", "plant_bushDetailed", "mushroom_red", "mushroom_tan");
                    biome.rockPrefabs = Pick(prefabs, "rock_smallB");
                    SetMicroRules(biome, prefabs, 1.2f,
                        Rule("Forest Canopy", "tree_oak", BiomeObjectCategory.Tree, 22f, 0.07f, 0.75f, 0f, 32f, 0.85f, 1.35f, 15f),
                        Rule("Forest Pines", "tree_pineTallA", BiomeObjectCategory.Tree, 10f, 0.1f, 0.78f, 0f, 30f, 0.8f, 1.25f, 18f),
                        Rule("Forest Underbrush", "plant_bushDetailed", BiomeObjectCategory.Detail, 42f, 0.05f, 0.72f, 0f, 30f, 0.8f, 1.25f, 4f),
                        Rule("Forest Mushrooms", "mushroom_red", BiomeObjectCategory.Detail, 18f, 0.05f, 0.7f, 0f, 22f, 0.65f, 1.15f, 2.2f));
                    break;
                case "meadow":
                    biome.treePrefabs = Pick(prefabs, "tree_small");
                    biome.detailPrefabs = Pick(prefabs, "grass", "grass_large", "flower_yellowA", "flower_redA", "flower_purpleA");
                    biome.rockPrefabs = Pick(prefabs, "rock_smallC");
                    SetMicroRules(biome, prefabs, 1.15f,
                        Rule("Meadow Grass", "grass", BiomeObjectCategory.Detail, 130f, 0.04f, 0.65f, 0f, 20f, 0.7f, 1.15f, 1f),
                        Rule("Meadow Flowers", "flower_yellowB", BiomeObjectCategory.Detail, 42f, 0.04f, 0.62f, 0f, 18f, 0.75f, 1.2f, 2f),
                        Rule("Small Meadow Trees", "tree_small", BiomeObjectCategory.Tree, 4f, 0.06f, 0.62f, 0f, 16f, 0.85f, 1.2f, 24f));
                    break;
                case "sparsegrass":
                    biome.treePrefabs = Pick(prefabs, "tree_small", "tree_cone");
                    biome.detailPrefabs = Pick(prefabs, "grass_leafs", "plant_flatShort", "plant_bushSmall");
                    biome.rockPrefabs = Pick(prefabs, "rock_smallB");
                    SetMicroRules(biome, prefabs, 0.8f,
                        Rule("Sparse Grass Tufts", "grass_leafs", BiomeObjectCategory.Detail, 70f, 0.04f, 0.68f, 0f, 24f, 0.75f, 1.2f, 1.8f),
                        Rule("Sparse Bushes", "plant_bushSmall", BiomeObjectCategory.Detail, 16f, 0.04f, 0.68f, 0f, 28f, 0.8f, 1.25f, 6f));
                    break;
                case "beach":
                    biome.treePrefabs = Pick(prefabs, "tree_palmTall", "tree_palmShort");
                    biome.detailPrefabs = Pick(prefabs, "log", "plant_flatShort");
                    biome.rockPrefabs = Pick(prefabs, "rock_smallA", "stone_smallA");
                    SetMicroRules(biome, prefabs, 0.75f,
                        Rule("Beach Palms", "tree_palmTall", BiomeObjectCategory.Tree, 4f, 0.02f, 0.38f, 0f, 16f, 0.85f, 1.25f, 25f),
                        Rule("Beach Driftwood", "log", BiomeObjectCategory.Prop, 8f, 0.02f, 0.35f, 0f, 12f, 0.85f, 1.2f, 12f),
                        Rule("Beach Stones", "stone_smallA", BiomeObjectCategory.Rock, 12f, 0.02f, 0.38f, 0f, 18f, 0.75f, 1.3f, 8f));
                    break;
                case "dunes":
                    biome.treePrefabs = new GameObject[0];
                    biome.detailPrefabs = Pick(prefabs, "cactus_short", "cactus_tall", "plant_flatTall");
                    biome.rockPrefabs = Pick(prefabs, "rock_smallA", "stone_smallB");
                    SetMicroRules(biome, prefabs, 0.65f,
                        Rule("Dune Cactus", "cactus_tall", BiomeObjectCategory.Detail, 9f, 0.04f, 0.5f, 0f, 20f, 0.75f, 1.25f, 18f),
                        Rule("Dune Stones", "stone_smallB", BiomeObjectCategory.Rock, 12f, 0.04f, 0.5f, 0f, 25f, 0.7f, 1.35f, 9f));
                    break;
                case "drydirt":
                    biome.treePrefabs = Pick(prefabs, "tree_cone");
                    biome.detailPrefabs = Pick(prefabs, "cactus_short", "plant_bushSmall", "plant_flatShort");
                    biome.rockPrefabs = Pick(prefabs, "rock_smallB", "stone_smallB");
                    SetMicroRules(biome, prefabs, 0.75f,
                        Rule("Dry Dirt Cactus", "cactus_short", BiomeObjectCategory.Detail, 15f, 0.08f, 0.68f, 0f, 30f, 0.75f, 1.25f, 12f),
                        Rule("Dry Dirt Rocks", "rock_smallB", BiomeObjectCategory.Rock, 20f, 0.08f, 0.72f, 0f, 38f, 0.75f, 1.45f, 8f));
                    break;
                case "mud":
                    biome.treePrefabs = new GameObject[0];
                    biome.detailPrefabs = Pick(prefabs, "plant_bushSmall", "grass_leafs", "log");
                    biome.rockPrefabs = Pick(prefabs, "stone_smallA");
                    SetMicroRules(biome, prefabs, 0.7f,
                        Rule("Mud Logs", "log_large", BiomeObjectCategory.Prop, 8f, 0.02f, 0.45f, 0f, 18f, 0.8f, 1.25f, 12f),
                        Rule("Mud Bushes", "plant_bushSmall", BiomeObjectCategory.Detail, 22f, 0.02f, 0.48f, 0f, 22f, 0.75f, 1.2f, 5f));
                    break;
                case "gravel":
                    biome.treePrefabs = new GameObject[0];
                    biome.detailPrefabs = Pick(prefabs, "plant_flatShort");
                    biome.rockPrefabs = Pick(prefabs, "rock_smallA", "rock_smallB", "stone_smallA", "stone_smallB");
                    SetMicroRules(biome, prefabs, 1f,
                        Rule("Gravel Stones", "stone_smallA", BiomeObjectCategory.Rock, 50f, 0.05f, 0.72f, 0f, 40f, 0.65f, 1.35f, 3.5f),
                        Rule("Gravel Rocks", "rock_smallC", BiomeObjectCategory.Rock, 22f, 0.05f, 0.76f, 2f, 48f, 0.65f, 1.45f, 6f));
                    break;
                case "barerock":
                case "alpinerock":
                case "highmountain":
                    biome.treePrefabs = Pick(prefabs, "tree_pineTallB");
                    biome.detailPrefabs = new GameObject[0];
                    biome.rockPrefabs = Pick(prefabs, "rock_largeA", "rock_largeB", "stone_tallA", "stone_largeA");
                    SetMicroRules(biome, prefabs, 1f,
                        Rule("Large Rock Scatter", "rock_largeB", BiomeObjectCategory.Rock, 30f, 0.28f, 1f, 12f, 78f, 0.55f, 1.55f, 8f),
                        Rule("Tall Stone Scatter", "stone_tallA", BiomeObjectCategory.Rock, 14f, 0.3f, 1f, 12f, 80f, 0.55f, 1.35f, 10f),
                        Rule("Sparse Alpine Pine", "tree_pineTallB", BiomeObjectCategory.Tree, 2f, 0.25f, 0.78f, 0f, 26f, 0.75f, 1.15f, 28f));
                    break;
                case "freshsnow":
                case "ice":
                    biome.treePrefabs = new GameObject[0];
                    biome.detailPrefabs = new GameObject[0];
                    biome.rockPrefabs = Pick(prefabs, "stone_smallB", "rock_smallC");
                    SetMicroRules(biome, prefabs, 0.45f,
                        Rule("Snow Stones", "stone_smallB", BiomeObjectCategory.Rock, 10f, 0.25f, 1f, 2f, 46f, 0.6f, 1.25f, 9f));
                    break;
                default:
                    SetMicroRules(biome, prefabs, 0.7f);
                    break;
            }
        }

        private static void SetMicroRules(BiomeDefinition biome, Dictionary<string, GameObject> prefabs, float densityMultiplier, params RuleSpec[] specs)
        {
            biome.placement.densityMultiplier = densityMultiplier;
            biome.placement.objectRules = BuildRules(prefabs, specs);
        }

        private static BiomeObjectPlacementRule[] BuildRules(Dictionary<string, GameObject> prefabs, params RuleSpec[] specs)
        {
            List<BiomeObjectPlacementRule> rules = new List<BiomeObjectPlacementRule>();
            foreach (RuleSpec spec in specs)
            {
                if (!prefabs.TryGetValue(spec.PrefabName, out GameObject prefab) || prefab == null)
                {
                    continue;
                }

                rules.Add(new BiomeObjectPlacementRule
                {
                    ruleName = spec.RuleName,
                    enabled = true,
                    category = spec.Category,
                    prefab = prefab,
                    densityPerSquareKm = spec.DensityPerSquareKm,
                    normalizedHeightRange = new Vector2(spec.MinHeight, spec.MaxHeight),
                    slopeDegreesRange = new Vector2(spec.MinSlope, spec.MaxSlope),
                    uniformScaleRange = new Vector2(spec.MinScale, spec.MaxScale),
                    minSpacingMeters = spec.MinSpacingMeters,
                    noiseScale = spec.Category == BiomeObjectCategory.Detail ? 3.5f : 1.25f,
                    noiseInfluence = spec.Category == BiomeObjectCategory.Tree ? 0.55f : 0.35f,
                    verticalOffsetMeters = Vector2.zero,
                    alignToTerrainNormal = spec.Category != BiomeObjectCategory.Tree,
                    normalAlignment = spec.Category == BiomeObjectCategory.Tree ? 0.35f : 0.85f,
                    randomYaw = 1f,
                    placementJitter = 0.85f
                });
            }

            return rules.ToArray();
        }

        private static RuleSpec Rule(
            string ruleName,
            string prefabName,
            BiomeObjectCategory category,
            float densityPerSquareKm,
            float minHeight,
            float maxHeight,
            float minSlope,
            float maxSlope,
            float minScale,
            float maxScale,
            float minSpacingMeters)
        {
            return new RuleSpec(ruleName, prefabName, category, densityPerSquareKm, minHeight, maxHeight, minSlope, maxSlope, minScale, maxScale, minSpacingMeters);
        }

        private static GameObject[] Pick(Dictionary<string, GameObject> prefabs, params string[] names)
        {
            List<GameObject> selected = new List<GameObject>();
            foreach (string name in names)
            {
                if (prefabs.TryGetValue(name, out GameObject prefab) && prefab != null)
                {
                    selected.Add(prefab);
                }
            }

            return selected.ToArray();
        }

        private static void ConfigureDefaultProfile()
        {
            TerrainGANPreviewProfile profile = AssetDatabase.LoadAssetAtPath<TerrainGANPreviewProfile>(ProfilePath);
            if (profile == null)
            {
                return;
            }

            Undo.RecordObject(profile, "Configure Terrain GAN object placement");
            profile.enableObjectPlacement = true;
            profile.clearObjectsWhenDisabled = true;
            profile.useObjectRulesPlacement = true;
            profile.useLegacyShortcutPlacement = false;
            profile.objectRulePlacementResolution = 128;
            profile.objectRuleDensityScale = 0.9f;
            profile.maxObjectRuleInstances = 1200;
            EditorUtility.SetDirty(profile);
        }

        private static string Normalize(string text)
        {
            return string.IsNullOrWhiteSpace(text)
                ? string.Empty
                : text.Replace(" ", string.Empty).Replace("_", string.Empty).Replace("-", string.Empty).ToLowerInvariant();
        }

        private static void EnsureFolder(string folder)
        {
            if (AssetDatabase.IsValidFolder(folder))
            {
                return;
            }

            string[] parts = folder.Split('/');
            string current = parts[0];
            for (int index = 1; index < parts.Length; index++)
            {
                string next = current + "/" + parts[index];
                if (!AssetDatabase.IsValidFolder(next))
                {
                    AssetDatabase.CreateFolder(current, parts[index]);
                }

                current = next;
            }
        }

        private readonly struct RuleSpec
        {
            public RuleSpec(
                string ruleName,
                string prefabName,
                BiomeObjectCategory category,
                float densityPerSquareKm,
                float minHeight,
                float maxHeight,
                float minSlope,
                float maxSlope,
                float minScale,
                float maxScale,
                float minSpacingMeters)
            {
                RuleName = ruleName;
                PrefabName = prefabName;
                Category = category;
                DensityPerSquareKm = densityPerSquareKm;
                MinHeight = minHeight;
                MaxHeight = maxHeight;
                MinSlope = minSlope;
                MaxSlope = maxSlope;
                MinScale = minScale;
                MaxScale = maxScale;
                MinSpacingMeters = minSpacingMeters;
            }

            public string RuleName { get; }
            public string PrefabName { get; }
            public BiomeObjectCategory Category { get; }
            public float DensityPerSquareKm { get; }
            public float MinHeight { get; }
            public float MaxHeight { get; }
            public float MinSlope { get; }
            public float MaxSlope { get; }
            public float MinScale { get; }
            public float MaxScale { get; }
            public float MinSpacingMeters { get; }
        }
    }
}
#endif
