using System;
using UnityEngine;

namespace UnityTerrainGANerator.Biomes
{
    [Serializable]
    public sealed class MacroBiomeSettings
    {
        public string macroName = "Macro Biome";
        public int macroClassIndex;
        public Color32 previewColor = new Color32(128, 128, 128, 255);
        public BiomePlacementSettings placement = new BiomePlacementSettings();
    }

    [CreateAssetMenu(menuName = "UnityTerrainGANerator/Biome Palette")]
    public sealed class BiomePalette : ScriptableObject
    {
        [Header("Macrobiomes Sent To Model")]
        public int macroClassCount = 7;
        public MacroBiomeSettings[] macroBiomes = CreateDefaultMacroBiomes();

        [Header("Microbiomes Painted By User")]
        public BiomeDefinition[] biomes;
        public int colorTolerance = 8;
        public int fallbackMacroClassIndex = 2;

        private void OnValidate()
        {
            macroClassCount = Mathf.Max(1, macroClassCount);
            fallbackMacroClassIndex = Mathf.Clamp(fallbackMacroClassIndex, 0, macroClassCount - 1);
            EnsureMacroBiomes();
        }

        public void EnsureMacroBiomes()
        {
            MacroBiomeSettings[] defaults = CreateDefaultMacroBiomes(macroClassCount);
            if (macroBiomes == null || macroBiomes.Length != macroClassCount)
            {
                MacroBiomeSettings[] next = new MacroBiomeSettings[macroClassCount];
                for (int index = 0; index < next.Length; index++)
                {
                    next[index] = index < (macroBiomes?.Length ?? 0) && macroBiomes[index] != null
                        ? macroBiomes[index]
                        : defaults[index];
                    next[index].macroClassIndex = index;
                }

                macroBiomes = next;
                return;
            }

            for (int index = 0; index < macroBiomes.Length; index++)
            {
                if (macroBiomes[index] == null)
                {
                    macroBiomes[index] = defaults[index];
                }

                macroBiomes[index].macroClassIndex = index;
            }
        }

        public BiomeDefinition FindNearestBiome(Color32 color)
        {
            if (biomes == null || biomes.Length == 0)
            {
                return null;
            }

            BiomeDefinition nearest = null;
            int nearestDistance = int.MaxValue;
            foreach (BiomeDefinition biome in biomes)
            {
                if (biome == null)
                {
                    continue;
                }

                int distance = SquaredDistance(color, biome.inputColor);
                if (distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearest = biome;
                }
            }

            int toleranceSquared = colorTolerance * colorTolerance;
            if (nearest == null || nearestDistance > toleranceSquared)
            {
                return null;
            }

            return nearest;
        }

        public int FindMacroClassIndex(Color32 color)
        {
            BiomeDefinition biome = FindNearestBiome(color);
            return ResolveMacroClassIndex(biome);
        }

        public int FindMicroBiomeIndex(Color32 color)
        {
            BiomeDefinition biome = FindNearestBiome(color);
            return biome == null ? -1 : biome.microBiomeId;
        }

        public BiomeDefinition FindBiomeByMicroBiomeId(int microBiomeId)
        {
            if (biomes == null)
            {
                return null;
            }

            foreach (BiomeDefinition biome in biomes)
            {
                if (biome != null && biome.microBiomeId == microBiomeId)
                {
                    return biome;
                }
            }

            return null;
        }

        public BiomeDefinition FindBiomeForMacroClass(int macroClassIndex)
        {
            if (biomes == null)
            {
                return null;
            }

            int resolvedMacro = Mathf.Clamp(macroClassIndex, 0, macroClassCount - 1);
            BiomeDefinition firstMatch = null;
            foreach (BiomeDefinition biome in biomes)
            {
                if (biome == null || Mathf.Clamp(biome.macroClassIndex, 0, macroClassCount - 1) != resolvedMacro)
                {
                    continue;
                }

                if (firstMatch == null)
                {
                    firstMatch = biome;
                }
                if (biome.microBiomeId == resolvedMacro)
                {
                    return biome;
                }
            }

            return firstMatch;
        }

        public BiomeDefinition FindFallbackBiome()
        {
            return FindBiomeForMacroClass(fallbackMacroClassIndex);
        }

        public MacroBiomeSettings GetMacroBiomeSettings(int macroClassIndex)
        {
            EnsureMacroBiomes();
            int resolvedMacro = Mathf.Clamp(macroClassIndex, 0, macroClassCount - 1);
            return macroBiomes == null || resolvedMacro >= macroBiomes.Length ? null : macroBiomes[resolvedMacro];
        }

        public Color32 FindMacroColor(int macroClassIndex)
        {
            BiomeDefinition biome = FindBiomeForMacroClass(macroClassIndex);
            if (biome != null)
            {
                return biome.MacroColor;
            }

            MacroBiomeSettings settings = GetMacroBiomeSettings(macroClassIndex);
            return settings == null ? new Color32(0, 0, 0, 255) : settings.previewColor;
        }

        public Color32 ConvertToMacroColor(Color32 color)
        {
            int macroClassIndex = FindMacroClassIndex(color);
            return FindMacroColor(macroClassIndex);
        }

        public int ResolveMacroClassIndex(BiomeDefinition biome)
        {
            if (biome == null)
            {
                return Mathf.Clamp(fallbackMacroClassIndex, 0, macroClassCount - 1);
            }

            return Mathf.Clamp(biome.macroClassIndex, 0, macroClassCount - 1);
        }

        public BiomeDefinition ResolveBiome(int microBiomeId, int macroClassIndex)
        {
            BiomeDefinition biome = FindBiomeByMicroBiomeId(microBiomeId);
            if (biome != null)
            {
                return biome;
            }

            biome = FindBiomeForMacroClass(macroClassIndex);
            return biome != null ? biome : FindFallbackBiome();
        }

        public static MacroBiomeSettings[] CreateDefaultMacroBiomes(int count = 7)
        {
            string[] names =
            {
                "Water",
                "Lowland",
                "Plain",
                "Hill",
                "Highland",
                "Mountain",
                "Peak"
            };

            Color32[] colors =
            {
                new Color32(53, 119, 191, 255),
                new Color32(101, 153, 84, 255),
                new Color32(172, 174, 103, 255),
                new Color32(169, 139, 91, 255),
                new Color32(130, 117, 103, 255),
                new Color32(101, 98, 98, 255),
                new Color32(224, 224, 224, 255)
            };

            int length = Mathf.Max(1, count);
            MacroBiomeSettings[] defaults = new MacroBiomeSettings[length];
            for (int index = 0; index < defaults.Length; index++)
            {
                int sourceIndex = Mathf.Min(index, names.Length - 1);
                defaults[index] = new MacroBiomeSettings
                {
                    macroName = index < names.Length ? names[index] : $"Macro {index}",
                    macroClassIndex = index,
                    previewColor = colors[sourceIndex]
                };
            }

            return defaults;
        }

        private static int SquaredDistance(Color32 a, Color32 b)
        {
            int dr = a.r - b.r;
            int dg = a.g - b.g;
            int db = a.b - b.b;
            return dr * dr + dg * dg + db * db;
        }
    }
}
