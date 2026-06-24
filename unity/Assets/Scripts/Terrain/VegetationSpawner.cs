using System.Collections.Generic;
using UnityEngine;
using UnityTerrainGANerator.Biomes;

namespace UnityTerrainGANerator.TerrainGeneration
{
    public sealed class VegetationSpawner : MonoBehaviour
    {
        private const string SpawnedRootName = "Generated Micro Biome Objects";

        public Terrain targetTerrain;
        public BiomePalette palette;
        public Transform spawnedRoot;
        public int seed = 1234;
        public bool clearPreviousObjects = true;

        [Header("Legacy Shortcut Placement")]
        public bool rebuildTerrainTrees = true;
        public bool spawnDetailPrefabs = true;
        public bool spawnRockPrefabs = true;
        [Range(8, 512)] public int placementResolution = 128;
        [Min(0f)] public float treeDensityScale = 0.12f;
        [Min(0f)] public float detailDensityScale = 0.18f;
        [Min(0f)] public float rockDensityScale = 0.06f;
        [Min(0)] public int maxTreeInstances = 2500;
        [Min(0)] public int maxDetailObjects = 2500;
        [Min(0)] public int maxRockObjects = 800;
        public bool randomYaw = true;

        [Header("Macro/Microbiome Object Rules")]
        public bool spawnPlacementRules = true;
        public bool spawnMacroPlacementRules = true;
        public bool spawnMicroPlacementRules = true;
        [Range(8, 512)] public int rulePlacementResolution = 128;
        [Min(0f)] public float ruleDensityScale = 1f;
        [Min(0)] public int maxRuleObjects = 3500;

        public void Spawn(MicroBiomeMap microBiomeMap)
        {
            Validate(microBiomeMap);

            if (clearPreviousObjects)
            {
                ClearSpawnedObjects();
            }

            if (rebuildTerrainTrees)
            {
                BuildTreeInstances(microBiomeMap);
            }

            if (spawnDetailPrefabs)
            {
                SpawnPrefabObjects(microBiomeMap, PlacementCategory.Detail, maxDetailObjects);
            }

            if (spawnRockPrefabs)
            {
                SpawnPrefabObjects(microBiomeMap, PlacementCategory.Rock, maxRockObjects);
            }

            if (spawnPlacementRules)
            {
                SpawnPlacementRuleObjects(microBiomeMap);
            }
        }

        public void BuildDetailLayers(MicroBiomeMap microBiomeMap)
        {
            Validate(microBiomeMap);
            SpawnPrefabObjects(microBiomeMap, PlacementCategory.Detail, maxDetailObjects);
        }

        public void SpawnPlacementRulesOnly(MicroBiomeMap microBiomeMap)
        {
            Validate(microBiomeMap);
            if (clearPreviousObjects)
            {
                ClearSpawnedObjects();
            }

            SpawnPlacementRuleObjects(microBiomeMap);
        }

        public void BuildTreeInstances(MicroBiomeMap microBiomeMap)
        {
            Validate(microBiomeMap);

            TerrainData data = targetTerrain.terrainData;
            Dictionary<GameObject, int> prefabToPrototype = new Dictionary<GameObject, int>();
            List<TreePrototype> prototypes = new List<TreePrototype>();
            RegisterTreePrototypes(prefabToPrototype, prototypes);

            if (prototypes.Count == 0 || maxTreeInstances == 0)
            {
                data.treeInstances = new TreeInstance[0];
                return;
            }

            data.treePrototypes = prototypes.ToArray();
            List<PlacementSample> samples = CollectSamples(microBiomeMap, PlacementCategory.Tree, maxTreeInstances);
            List<TreeInstance> treeInstances = new List<TreeInstance>(samples.Count);

            foreach (PlacementSample sample in samples)
            {
                GameObject[] prefabs = sample.Biome.treePrefabs;
                GameObject prefab = SelectPrefab(prefabs, sample.GridX, sample.GridY, sample.Salt + 13);
                if (prefab == null || !prefabToPrototype.TryGetValue(prefab, out int prototypeIndex))
                {
                    continue;
                }

                float scale = SelectScale(sample.Biome.treeScaleRange, sample.GridX, sample.GridY, sample.Salt + 29);
                TreeInstance tree = new TreeInstance
                {
                    prototypeIndex = prototypeIndex,
                    position = new Vector3(sample.U, sample.NormalizedHeight, sample.V),
                    widthScale = scale,
                    heightScale = scale,
                    color = Color.white,
                    lightmapColor = Color.white,
                    rotation = randomYaw ? Hash01(sample.GridX, sample.GridY, sample.Salt + 31) * Mathf.PI * 2f : 0f
                };
                treeInstances.Add(tree);
            }

            data.treeInstances = treeInstances.ToArray();
            targetTerrain.Flush();
        }

        private void SpawnPrefabObjects(MicroBiomeMap microBiomeMap, PlacementCategory category, int maxObjects)
        {
            if (maxObjects == 0)
            {
                return;
            }

            Transform root = ResolveSpawnRoot();
            List<PlacementSample> samples = CollectSamples(microBiomeMap, category, maxObjects);

            foreach (PlacementSample sample in samples)
            {
                GameObject prefab = SelectPrefab(GetPrefabs(sample.Biome, category), sample.GridX, sample.GridY, sample.Salt + 47);
                if (prefab == null)
                {
                    continue;
                }

                Quaternion yawRotation = randomYaw
                    ? Quaternion.Euler(0f, Hash01(sample.GridX, sample.GridY, sample.Salt + 53) * 360f, 0f)
                    : Quaternion.identity;
                Quaternion rotation = sample.Biome.alignObjectsToTerrainNormal
                    ? Quaternion.Slerp(Quaternion.identity, Quaternion.FromToRotation(Vector3.up, sample.Normal), sample.Biome.normalAlignment) * yawRotation
                    : yawRotation;

                GameObject instance = Instantiate(prefab, sample.WorldPosition, rotation, root);
                float scale = SelectScale(sample.Biome.objectScaleRange, sample.GridX, sample.GridY, sample.Salt + 59);
                instance.transform.localScale = Vector3.Scale(instance.transform.localScale, Vector3.one * scale);
            }
        }

        private void SpawnPlacementRuleObjects(MicroBiomeMap microBiomeMap)
        {
            if (maxRuleObjects == 0 || (!spawnMacroPlacementRules && !spawnMicroPlacementRules))
            {
                return;
            }

            TerrainData data = targetTerrain.terrainData;
            Transform root = ResolveSpawnRoot();
            int resolution = Mathf.Max(1, rulePlacementResolution);
            float cellSize = 1f / resolution;
            float terrainAreaSquareKm = Mathf.Max(0.000001f, (data.size.x * data.size.z) / 1000000f);
            float cellAreaSquareKm = terrainAreaSquareKm / (resolution * resolution);
            int spawnedCount = 0;
            List<RuleSpacingRecord> spacingRecords = new List<RuleSpacingRecord>(Mathf.Min(maxRuleObjects, 512));

            for (int y = 0; y < resolution; y++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    float baseU = (x + 0.5f) * cellSize;
                    float baseV = (y + 0.5f) * cellSize;
                    int macroClass = microBiomeMap.SampleMacroClassIndex(baseU, baseV);
                    int microBiome = microBiomeMap.SampleMicroBiomeIndex(baseU, baseV);
                    BiomeDefinition biome = palette.ResolveBiome(microBiome, macroClass);

                    if (spawnMacroPlacementRules && (biome == null || microBiome < 0 || biome.inheritMacroPlacement))
                    {
                        MacroBiomeSettings macroSettings = palette.GetMacroBiomeSettings(macroClass);
                        if (macroSettings != null)
                        {
                            spawnedCount += SpawnRulesFromSettings(
                                macroSettings.placement,
                                microBiomeMap,
                                data,
                                root,
                                spacingRecords,
                                x,
                                y,
                                baseU,
                                baseV,
                                cellSize,
                                cellAreaSquareKm,
                                macroClass,
                                microBiome,
                                false,
                                3000001 + macroClass * 9176,
                                maxRuleObjects - spawnedCount);
                        }
                    }

                    if (spawnMicroPlacementRules && biome != null && microBiome >= 0)
                    {
                        spawnedCount += SpawnRulesFromSettings(
                            biome.placement,
                            microBiomeMap,
                            data,
                            root,
                            spacingRecords,
                            x,
                            y,
                            baseU,
                            baseV,
                            cellSize,
                            cellAreaSquareKm,
                            macroClass,
                            microBiome,
                            true,
                            biome.seedOffset + 7000003,
                            maxRuleObjects - spawnedCount);
                    }

                    if (spawnedCount >= maxRuleObjects)
                    {
                        targetTerrain.Flush();
                        return;
                    }
                }
            }

            targetTerrain.Flush();
        }

        private int SpawnRulesFromSettings(
            BiomePlacementSettings settings,
            MicroBiomeMap microBiomeMap,
            TerrainData data,
            Transform root,
            List<RuleSpacingRecord> spacingRecords,
            int gridX,
            int gridY,
            float baseU,
            float baseV,
            float cellSize,
            float cellAreaSquareKm,
            int macroClass,
            int microBiome,
            bool requireMicroMatch,
            int ownerSalt,
            int remainingBudget)
        {
            if (remainingBudget <= 0 || settings == null || settings.objectRules == null || settings.objectRules.Length == 0)
            {
                return 0;
            }

            int spawned = 0;
            float densityMultiplier = Mathf.Max(0f, settings.densityMultiplier);
            if (densityMultiplier <= 0f)
            {
                return 0;
            }

            for (int ruleIndex = 0; ruleIndex < settings.objectRules.Length; ruleIndex++)
            {
                BiomeObjectPlacementRule rule = settings.objectRules[ruleIndex];
                if (rule == null || !rule.enabled || rule.prefab == null || rule.densityPerSquareKm <= 0f)
                {
                    continue;
                }

                int salt = seed + ownerSalt + ruleIndex * 1000003 + (int)rule.category * 8191;
                float jitter = Mathf.Clamp01(rule.placementJitter) * cellSize;
                float u = Mathf.Clamp01(baseU + (Hash01(gridX, gridY, salt + 1) - 0.5f) * jitter);
                float v = Mathf.Clamp01(baseV + (Hash01(gridX, gridY, salt + 2) - 0.5f) * jitter);

                int sampleMacroClass = microBiomeMap.SampleMacroClassIndex(u, v);
                if (sampleMacroClass != macroClass)
                {
                    continue;
                }

                if (requireMicroMatch && microBiomeMap.SampleMicroBiomeIndex(u, v) != microBiome)
                {
                    continue;
                }

                float terrainHeight = data.GetInterpolatedHeight(u, v);
                float normalizedHeight = Mathf.Clamp01(terrainHeight / Mathf.Max(data.size.y, 0.0001f));
                float slope = data.GetSteepness(u, v);
                if (!Accepts(settings.normalizedHeightRange, normalizedHeight) ||
                    !Accepts(settings.slopeDegreesRange, slope) ||
                    !Accepts(rule.normalizedHeightRange, normalizedHeight) ||
                    !Accepts(rule.slopeDegreesRange, slope))
                {
                    continue;
                }

                float expectedInCell = rule.densityPerSquareKm * densityMultiplier * ruleDensityScale * cellAreaSquareKm;
                if (expectedInCell <= 0f)
                {
                    continue;
                }

                float noise = SampleRulePlacementNoise(rule, u, v, salt + 3);
                float influence = Mathf.Clamp01(rule.noiseInfluence);
                float probability = Mathf.Clamp01(expectedInCell * Mathf.Lerp(1f - influence, 1f + influence, noise));
                if (Hash01(gridX, gridY, salt + 4) > probability)
                {
                    continue;
                }

                Vector3 localPosition = new Vector3(u * data.size.x, terrainHeight, v * data.size.z);
                Vector3 worldPosition = targetTerrain.transform.TransformPoint(localPosition);
                Vector3 normal = targetTerrain.transform.TransformDirection(data.GetInterpolatedNormal(u, v)).normalized;
                float minSpacing = Mathf.Max(0f, rule.minSpacingMeters);
                if (!PassesSpacing(spacingRecords, worldPosition, minSpacing))
                {
                    continue;
                }

                SpawnRuleObject(rule, root, worldPosition, normal, gridX, gridY, salt);
                spacingRecords.Add(new RuleSpacingRecord(worldPosition, minSpacing));
                spawned++;

                if (spawned >= remainingBudget)
                {
                    break;
                }
            }

            return spawned;
        }

        private void SpawnRuleObject(
            BiomeObjectPlacementRule rule,
            Transform root,
            Vector3 worldPosition,
            Vector3 normal,
            int gridX,
            int gridY,
            int salt)
        {
            float verticalOffset = SelectSignedRange(rule.verticalOffsetMeters, gridX, gridY, salt + 5);
            Vector3 offsetDirection = rule.alignToTerrainNormal ? normal : Vector3.up;
            Vector3 position = worldPosition + offsetDirection * verticalOffset;
            float yaw = randomYaw ? Hash01(gridX, gridY, salt + 6) * 360f * Mathf.Clamp01(rule.randomYaw) : 0f;
            Quaternion yawRotation = Quaternion.Euler(0f, yaw, 0f);
            Quaternion surfaceRotation = rule.alignToTerrainNormal
                ? Quaternion.Slerp(
                    Quaternion.identity,
                    Quaternion.FromToRotation(Vector3.up, normal),
                    Mathf.Clamp01(rule.normalAlignment))
                : Quaternion.identity;
            Quaternion rotation = surfaceRotation * yawRotation;

            GameObject instance = Instantiate(rule.prefab, position, rotation, root);
            instance.name = string.IsNullOrWhiteSpace(rule.ruleName)
                ? rule.prefab.name
                : $"{rule.ruleName} ({rule.category})";
            float scale = SelectScale(rule.uniformScaleRange, gridX, gridY, salt + 7);
            instance.transform.localScale = Vector3.Scale(instance.transform.localScale, Vector3.one * scale);
        }

        private List<PlacementSample> CollectSamples(MicroBiomeMap microBiomeMap, PlacementCategory category, int maxCount)
        {
            TerrainData data = targetTerrain.terrainData;
            int resolution = Mathf.Max(1, placementResolution);
            float cellSize = 1f / resolution;
            List<PlacementSample> samples = new List<PlacementSample>(Mathf.Min(maxCount, resolution * resolution));

            for (int y = 0; y < resolution; y++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    float baseU = (x + 0.5f) * cellSize;
                    float baseV = (y + 0.5f) * cellSize;
                    int macroClass = microBiomeMap.SampleMacroClassIndex(baseU, baseV);
                    int microBiome = microBiomeMap.SampleMicroBiomeIndex(baseU, baseV);
                    BiomeDefinition biome = palette.ResolveBiome(microBiome, macroClass);
                    if (biome == null || GetPrefabs(biome, category).Length == 0)
                    {
                        continue;
                    }

                    int salt = seed + biome.seedOffset + (int)category * 1000003;
                    float jitter = Mathf.Clamp01(biome.placementJitter) * cellSize;
                    float u = Mathf.Clamp01(baseU + (Hash01(x, y, salt + 1) - 0.5f) * jitter);
                    float v = Mathf.Clamp01(baseV + (Hash01(x, y, salt + 2) - 0.5f) * jitter);
                    float terrainHeight = data.GetInterpolatedHeight(u, v);
                    float normalizedHeight = Mathf.Clamp01(terrainHeight / Mathf.Max(data.size.y, 0.0001f));
                    float slope = data.GetSteepness(u, v);
                    if (!biome.Accepts(normalizedHeight, slope))
                    {
                        continue;
                    }

                    float density = Mathf.Clamp01(GetDensity(biome, category) * GetDensityScale(category));
                    if (density <= 0f)
                    {
                        continue;
                    }

                    float noise = SamplePlacementNoise(biome, u, v, salt + 3);
                    float probability = Mathf.Clamp01(density * Mathf.Lerp(0.35f, 1.35f, noise));
                    if (Hash01(x, y, salt + 4) > probability)
                    {
                        continue;
                    }

                    Vector3 localPosition = new Vector3(u * data.size.x, terrainHeight, v * data.size.z);
                    Vector3 worldPosition = targetTerrain.transform.TransformPoint(localPosition);
                    Vector3 normal = targetTerrain.transform.TransformDirection(data.GetInterpolatedNormal(u, v)).normalized;
                    samples.Add(new PlacementSample(biome, x, y, salt, u, v, normalizedHeight, worldPosition, normal));

                    if (samples.Count >= maxCount)
                    {
                        return samples;
                    }
                }
            }

            return samples;
        }

        private void RegisterTreePrototypes(Dictionary<GameObject, int> prefabToPrototype, List<TreePrototype> prototypes)
        {
            if (palette.biomes == null)
            {
                return;
            }

            foreach (BiomeDefinition biome in palette.biomes)
            {
                if (biome == null || biome.treePrefabs == null)
                {
                    continue;
                }

                foreach (GameObject prefab in biome.treePrefabs)
                {
                    if (prefab == null || prefabToPrototype.ContainsKey(prefab))
                    {
                        continue;
                    }

                    TreePrototype prototype = new TreePrototype
                    {
                        prefab = prefab,
                        bendFactor = 0.2f
                    };
                    prefabToPrototype.Add(prefab, prototypes.Count);
                    prototypes.Add(prototype);
                }
            }
        }

        private Transform ResolveSpawnRoot()
        {
            if (spawnedRoot != null)
            {
                return spawnedRoot;
            }

            Transform existingRoot = transform.Find(SpawnedRootName);
            if (existingRoot != null)
            {
                spawnedRoot = existingRoot;
                return spawnedRoot;
            }

            GameObject root = new GameObject(SpawnedRootName);
            root.transform.SetParent(transform, false);
            spawnedRoot = root.transform;
            return spawnedRoot;
        }

        public void ClearSpawnedObjects()
        {
            if (spawnedRoot == null)
            {
                spawnedRoot = transform.Find(SpawnedRootName);
            }

            if (spawnedRoot == null)
            {
                return;
            }

            for (int i = spawnedRoot.childCount - 1; i >= 0; i--)
            {
                Transform child = spawnedRoot.GetChild(i);
                if (Application.isPlaying)
                {
                    Destroy(child.gameObject);
                }
                else
                {
                    DestroyImmediate(child.gameObject);
                }
            }
        }

        private void Validate(MicroBiomeMap microBiomeMap)
        {
            if (microBiomeMap == null)
            {
                throw new System.ArgumentNullException(nameof(microBiomeMap));
            }

            if (targetTerrain == null)
            {
                targetTerrain = GetComponent<Terrain>();
            }

            if (targetTerrain == null)
            {
                throw new System.InvalidOperationException("Target Terrain is not assigned.");
            }

            if (palette == null)
            {
                throw new System.InvalidOperationException("Biome palette is not assigned.");
            }
        }

        private static bool Accepts(Vector2 range, float value)
        {
            float min = Mathf.Min(range.x, range.y);
            float max = Mathf.Max(range.x, range.y);
            return value >= min && value <= max;
        }

        private static bool PassesSpacing(List<RuleSpacingRecord> records, Vector3 worldPosition, float minSpacing)
        {
            if (minSpacing <= 0f)
            {
                return true;
            }

            Vector2 position = new Vector2(worldPosition.x, worldPosition.z);
            for (int index = 0; index < records.Count; index++)
            {
                RuleSpacingRecord record = records[index];
                float requiredSpacing = Mathf.Max(minSpacing, record.MinSpacingMeters);
                if (requiredSpacing <= 0f)
                {
                    continue;
                }

                Vector2 other = new Vector2(record.WorldPosition.x, record.WorldPosition.z);
                if ((position - other).sqrMagnitude < requiredSpacing * requiredSpacing)
                {
                    return false;
                }
            }

            return true;
        }

        private static GameObject[] GetPrefabs(BiomeDefinition biome, PlacementCategory category)
        {
            switch (category)
            {
                case PlacementCategory.Tree:
                    return biome.treePrefabs ?? new GameObject[0];
                case PlacementCategory.Detail:
                    return biome.detailPrefabs ?? new GameObject[0];
                case PlacementCategory.Rock:
                    return biome.rockPrefabs ?? new GameObject[0];
                default:
                    return new GameObject[0];
            }
        }

        private static float GetDensity(BiomeDefinition biome, PlacementCategory category)
        {
            switch (category)
            {
                case PlacementCategory.Tree:
                    return biome.treeDensity;
                case PlacementCategory.Detail:
                    return biome.detailDensity;
                case PlacementCategory.Rock:
                    return biome.rockDensity;
                default:
                    return 0f;
            }
        }

        private float GetDensityScale(PlacementCategory category)
        {
            switch (category)
            {
                case PlacementCategory.Tree:
                    return treeDensityScale;
                case PlacementCategory.Detail:
                    return detailDensityScale;
                case PlacementCategory.Rock:
                    return rockDensityScale;
                default:
                    return 0f;
            }
        }

        private static GameObject SelectPrefab(GameObject[] prefabs, int x, int y, int salt)
        {
            if (prefabs == null || prefabs.Length == 0)
            {
                return null;
            }

            int index = Mathf.Clamp(Mathf.FloorToInt(Hash01(x, y, salt) * prefabs.Length), 0, prefabs.Length - 1);
            return prefabs[index];
        }

        private static float SelectScale(Vector2 scaleRange, int x, int y, int salt)
        {
            float min = Mathf.Max(0.001f, Mathf.Min(scaleRange.x, scaleRange.y));
            float max = Mathf.Max(min, Mathf.Max(scaleRange.x, scaleRange.y));
            return Mathf.Lerp(min, max, Hash01(x, y, salt));
        }

        private static float SelectSignedRange(Vector2 range, int x, int y, int salt)
        {
            float min = Mathf.Min(range.x, range.y);
            float max = Mathf.Max(range.x, range.y);
            return Mathf.Lerp(min, max, Hash01(x, y, salt));
        }

        private static float SamplePlacementNoise(BiomeDefinition biome, float u, float v, int salt)
        {
            float scale = Mathf.Max(0.001f, biome.noiseScale);
            float offsetX = (salt % 9973) * 0.013f;
            float offsetY = (salt % 7919) * 0.017f;
            return Mathf.PerlinNoise(u * scale + offsetX, v * scale + offsetY);
        }

        private static float SampleRulePlacementNoise(BiomeObjectPlacementRule rule, float u, float v, int salt)
        {
            float scale = Mathf.Max(0.001f, rule.noiseScale);
            float offsetX = (salt % 9973) * 0.013f;
            float offsetY = (salt % 7919) * 0.017f;
            return Mathf.PerlinNoise(u * scale + offsetX, v * scale + offsetY);
        }

        private static float Hash01(int x, int y, int salt)
        {
            unchecked
            {
                uint hash = (uint)salt;
                hash ^= (uint)x * 374761393u;
                hash = (hash << 13) | (hash >> 19);
                hash ^= (uint)y * 668265263u;
                hash *= 2246822519u;
                hash ^= hash >> 15;
                hash *= 3266489917u;
                hash ^= hash >> 16;
                return (hash & 0x00FFFFFFu) / 16777215f;
            }
        }

        private enum PlacementCategory
        {
            Tree = 0,
            Detail = 1,
            Rock = 2
        }

        private readonly struct RuleSpacingRecord
        {
            public RuleSpacingRecord(Vector3 worldPosition, float minSpacingMeters)
            {
                WorldPosition = worldPosition;
                MinSpacingMeters = minSpacingMeters;
            }

            public Vector3 WorldPosition { get; }
            public float MinSpacingMeters { get; }
        }

        private readonly struct PlacementSample
        {
            public PlacementSample(
                BiomeDefinition biome,
                int gridX,
                int gridY,
                int salt,
                float u,
                float v,
                float normalizedHeight,
                Vector3 worldPosition,
                Vector3 normal)
            {
                Biome = biome;
                GridX = gridX;
                GridY = gridY;
                Salt = salt;
                U = u;
                V = v;
                NormalizedHeight = normalizedHeight;
                WorldPosition = worldPosition;
                Normal = normal;
            }

            public BiomeDefinition Biome { get; }
            public int GridX { get; }
            public int GridY { get; }
            public int Salt { get; }
            public float U { get; }
            public float V { get; }
            public float NormalizedHeight { get; }
            public Vector3 WorldPosition { get; }
            public Vector3 Normal { get; }
        }
    }
}
