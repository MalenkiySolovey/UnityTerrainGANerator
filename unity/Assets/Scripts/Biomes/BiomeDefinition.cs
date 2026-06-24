using System;
using UnityEngine;

namespace UnityTerrainGANerator.Biomes
{
    public enum BiomeObjectCategory
    {
        Tree = 0,
        Detail = 1,
        Rock = 2,
        Prop = 3
    }

    [Serializable]
    public sealed class BiomeObjectPlacementRule
    {
        public string ruleName = "Object Rule";
        public bool enabled = true;
        public BiomeObjectCategory category = BiomeObjectCategory.Prop;
        public GameObject prefab;
        [Min(0f)] public float densityPerSquareKm = 25f;
        public Vector2 normalizedHeightRange = new Vector2(0f, 1f);
        public Vector2 slopeDegreesRange = new Vector2(0f, 45f);
        public Vector2 uniformScaleRange = new Vector2(0.8f, 1.2f);
        [Min(0f)] public float minSpacingMeters;
        [Min(0.001f)] public float noiseScale = 1f;
        [Range(0f, 1f)] public float noiseInfluence = 0.35f;
        public Vector2 verticalOffsetMeters = Vector2.zero;
        public bool alignToTerrainNormal = true;
        [Range(0f, 1f)] public float normalAlignment = 0.85f;
        [Range(0f, 1f)] public float randomYaw = 1f;
        [Range(0f, 1f)] public float placementJitter = 0.35f;
    }

    [Serializable]
    public sealed class BiomePlacementSettings
    {
        public TerrainLayer[] terrainLayers;
        [Min(0f)] public float densityMultiplier = 1f;
        public Vector2 normalizedHeightRange = new Vector2(0f, 1f);
        public Vector2 slopeDegreesRange = new Vector2(0f, 90f);
        public BiomeObjectPlacementRule[] objectRules;
    }

    [CreateAssetMenu(menuName = "UnityTerrainGANerator/Biome Definition")]
    public sealed class BiomeDefinition : ScriptableObject
    {
        [Header("Identity")]
        public string biomeName = "New Microbiome";
        public int microBiomeId;
        public int macroClassIndex;
        public Color32 inputColor = new Color32(128, 128, 128, 255);
        public Color32 macroInputColor = new Color32(0, 0, 0, 0);

        [Header("Terrain Paint")]
        public TerrainLayer[] terrainLayers;
        public float[] terrainLayerWeights;

        [Header("Object Prefab Shortcuts")]
        public GameObject[] treePrefabs;
        public GameObject[] detailPrefabs;
        public GameObject[] rockPrefabs;

        [Header("Placement Filters")]
        public Vector2 heightRange = new Vector2(0f, 1f);
        public Vector2 slopeRange = new Vector2(0f, 90f);

        [Header("Placement Density")]
        [Min(0f)] public float treeDensity;
        [Min(0f)] public float detailDensity;
        [Min(0f)] public float rockDensity;

        [Header("Placement Variation")]
        [Min(0.001f)] public float noiseScale = 1f;
        public Vector2 treeScaleRange = new Vector2(0.85f, 1.25f);
        public Vector2 objectScaleRange = new Vector2(0.75f, 1.35f);
        [Range(0f, 1f)] public float placementJitter = 0.85f;
        public bool alignObjectsToTerrainNormal = true;
        [Range(0f, 1f)] public float normalAlignment = 0.85f;
        public int seedOffset;

        [Header("Detailed Placement Rules")]
        public bool inheritMacroPlacement = true;
        public BiomePlacementSettings placement = new BiomePlacementSettings();

        public Color32 MacroColor
        {
            get
            {
                return macroInputColor.a > 0 ? macroInputColor : inputColor;
            }
        }

        public bool Accepts(float normalizedHeight, float slopeDegrees)
        {
            float minHeight = Mathf.Min(heightRange.x, heightRange.y);
            float maxHeight = Mathf.Max(heightRange.x, heightRange.y);
            float minSlope = Mathf.Min(slopeRange.x, slopeRange.y);
            float maxSlope = Mathf.Max(slopeRange.x, slopeRange.y);
            return normalizedHeight >= minHeight
                && normalizedHeight <= maxHeight
                && slopeDegrees >= minSlope
                && slopeDegrees <= maxSlope;
        }

        public float GetTerrainLayerWeight(int index)
        {
            if (terrainLayerWeights == null || index < 0 || index >= terrainLayerWeights.Length)
            {
                return 1f;
            }

            return Mathf.Max(0f, terrainLayerWeights[index]);
        }
    }
}
