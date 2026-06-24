#if UNITY_EDITOR
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Biomes;

namespace UnityTerrainGANerator.Editor
{
    [CustomEditor(typeof(BiomePalette))]
    public sealed class BiomePaletteEditor : UnityEditor.Editor
    {
        private SerializedProperty macroClassCountProperty;
        private SerializedProperty macroBiomesProperty;
        private SerializedProperty biomesProperty;
        private SerializedProperty colorToleranceProperty;
        private SerializedProperty fallbackMacroClassIndexProperty;
        private static readonly Dictionary<int, bool> MicroBiomeObjectFoldouts = new Dictionary<int, bool>();

        private void OnEnable()
        {
            macroClassCountProperty = serializedObject.FindProperty("macroClassCount");
            macroBiomesProperty = serializedObject.FindProperty("macroBiomes");
            biomesProperty = serializedObject.FindProperty("biomes");
            colorToleranceProperty = serializedObject.FindProperty("colorTolerance");
            fallbackMacroClassIndexProperty = serializedObject.FindProperty("fallbackMacroClassIndex");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.LabelField("Model Macro Contract", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(macroClassCountProperty);
            EditorGUILayout.PropertyField(fallbackMacroClassIndexProperty);
            EditorGUILayout.PropertyField(colorToleranceProperty);
            EditorGUILayout.HelpBox("The ONNX model receives 7 macro channels. User-painted microbiomes stay editable in Unity, then map back to one macro class before inference.", MessageType.Info);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Macrobiomes", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox("Macrobiomes are the 7 model classes. Object rules here are shared defaults for all microbiomes that inherit macro placement.", MessageType.None);
            DrawMacroBiomePlacementEditor(macroBiomesProperty);
            if (GUILayout.Button("Reset v12.4 Macrobiomes"))
            {
                ResetMacrobiomes((BiomePalette)target);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Microbiomes", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(biomesProperty, true);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Create Microbiome Asset"))
            {
                CreateMicrobiomeAsset((BiomePalette)target);
            }

            if (GUILayout.Button("Create Default Microbiomes"))
            {
                TerrainGANDemoAssetCreator.CreateDefaultMicroBiomeAssets();
            }
            EditorGUILayout.EndHorizontal();

            serializedObject.ApplyModifiedProperties();
            DrawMicroBiomeObjectAssignment((BiomePalette)target);
            DrawValidation((BiomePalette)target);
        }

        public static void CreateMicrobiomeAsset(BiomePalette palette)
        {
            string folder = ResolvePaletteFolder(palette);
            string path = EditorUtility.SaveFilePanelInProject(
                "Create Microbiome",
                "New Microbiome.asset",
                "asset",
                "Choose where to save the microbiome asset.",
                folder);

            if (string.IsNullOrWhiteSpace(path))
            {
                return;
            }

            BiomeDefinition biome = ScriptableObject.CreateInstance<BiomeDefinition>();
            biome.biomeName = Path.GetFileNameWithoutExtension(path);
            biome.microBiomeId = FindNextMicroBiomeId(palette);
            int maxMacroClass = Mathf.Max(0, (palette != null ? palette.macroClassCount : 7) - 1);
            biome.macroClassIndex = Mathf.Clamp(palette != null ? palette.fallbackMacroClassIndex : 0, 0, maxMacroClass);
            biome.inputColor = NextColorForId(biome.microBiomeId);

            AssetDatabase.CreateAsset(biome, path);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            if (palette != null)
            {
                Undo.RecordObject(palette, "Add microbiome to palette");
                int oldLength = palette.biomes != null ? palette.biomes.Length : 0;
                BiomeDefinition[] next = new BiomeDefinition[oldLength + 1];
                for (int index = 0; index < oldLength; index++)
                {
                    next[index] = palette.biomes[index];
                }

                next[oldLength] = biome;
                palette.biomes = next;
                EditorUtility.SetDirty(palette);
            }

            Selection.activeObject = biome;
        }

        private static void ResetMacrobiomes(BiomePalette palette)
        {
            Undo.RecordObject(palette, "Reset v12.4 macrobiomes");
            palette.macroClassCount = 7;
            palette.macroBiomes = BiomePalette.CreateDefaultMacroBiomes();
            EditorUtility.SetDirty(palette);
        }

        private static void DrawMacroBiomePlacementEditor(SerializedProperty macroBiomes)
        {
            if (macroBiomes == null)
            {
                return;
            }

            for (int index = 0; index < macroBiomes.arraySize; index++)
            {
                SerializedProperty macro = macroBiomes.GetArrayElementAtIndex(index);
                SerializedProperty macroName = macro.FindPropertyRelative("macroName");
                string label = $"{index}: {(macroName != null ? macroName.stringValue : "Macro Biome")}";
                macro.isExpanded = EditorGUILayout.Foldout(macro.isExpanded, label, true);
                if (!macro.isExpanded)
                {
                    continue;
                }

                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                EditorGUILayout.PropertyField(macroName);
                EditorGUILayout.PropertyField(macro.FindPropertyRelative("previewColor"));
                using (new EditorGUI.DisabledScope(true))
                {
                    EditorGUILayout.PropertyField(macro.FindPropertyRelative("macroClassIndex"));
                }

                DrawPlacementSettings(
                    macro.FindPropertyRelative("placement"),
                    "Macro Object Rules",
                    BiomeObjectCategory.Prop,
                    "Drop shared trees, rocks, plants or props here. Microbiomes can inherit these defaults and then add their own local rules.");
                EditorGUILayout.EndVertical();
            }
        }

        private static void DrawMicroBiomeObjectAssignment(BiomePalette palette)
        {
            if (palette == null || palette.biomes == null || palette.biomes.Length == 0)
            {
                return;
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Microbiome Object Assignment", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox("Drag prefab assets into a microbiome. Tree/detail/rock shortcuts use the legacy VegetationSpawner passes; detailed rules use the area-based object-rule placement pass.", MessageType.Info);

            foreach (BiomeDefinition biome in palette.biomes)
            {
                if (biome == null)
                {
                    continue;
                }

                SerializedObject biomeObject = new SerializedObject(biome);
                biomeObject.Update();

                int id = biome.GetInstanceID();
                bool expanded = MicroBiomeObjectFoldouts.TryGetValue(id, out bool storedExpanded) ? storedExpanded : false;
                string macroName = ResolveMacroName(palette, biome.macroClassIndex);
                string title = $"{biome.microBiomeId}: {biome.biomeName} -> {macroName}";
                expanded = EditorGUILayout.Foldout(expanded, title, true);
                MicroBiomeObjectFoldouts[id] = expanded;
                if (!expanded)
                {
                    continue;
                }

                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                EditorGUILayout.PropertyField(biomeObject.FindProperty("macroClassIndex"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("inheritMacroPlacement"));

                BiomeObjectPlacementEditorUtility.DrawPrefabArrayDropZone(
                    biomeObject.FindProperty("treePrefabs"),
                    "Tree Prefabs",
                    "Drop tree prefab assets for this microbiome.");
                BiomeObjectPlacementEditorUtility.DrawPrefabArrayDropZone(
                    biomeObject.FindProperty("detailPrefabs"),
                    "Detail / Plant Prefabs",
                    "Drop grass, flowers, shrubs and other small detail prefab assets for this microbiome.");
                BiomeObjectPlacementEditorUtility.DrawPrefabArrayDropZone(
                    biomeObject.FindProperty("rockPrefabs"),
                    "Rock Prefabs",
                    "Drop rock, cliff and boulder prefab assets for this microbiome.");

                EditorGUILayout.LabelField("Shortcut Density And Filters", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(biomeObject.FindProperty("heightRange"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("slopeRange"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("treeDensity"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("detailDensity"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("rockDensity"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("noiseScale"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("treeScaleRange"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("objectScaleRange"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("placementJitter"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("alignObjectsToTerrainNormal"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("normalAlignment"));
                EditorGUILayout.PropertyField(biomeObject.FindProperty("seedOffset"));

                DrawPlacementSettings(
                    biomeObject.FindProperty("placement"),
                    "Microbiome Detailed Rules",
                    BiomeObjectCategory.Prop,
                    "Drop extra prefab assets here when one object needs its own density, height, slope or scale filters.");

                biomeObject.ApplyModifiedProperties();
                EditorGUILayout.EndVertical();
            }
        }

        private static void DrawPlacementSettings(
            SerializedProperty placement,
            string objectRulesTitle,
            BiomeObjectCategory defaultCategory,
            string helpText)
        {
            if (placement == null)
            {
                return;
            }

            EditorGUILayout.LabelField("Placement Settings", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(placement.FindPropertyRelative("terrainLayers"), true);
            EditorGUILayout.PropertyField(placement.FindPropertyRelative("densityMultiplier"));
            EditorGUILayout.PropertyField(placement.FindPropertyRelative("normalizedHeightRange"));
            EditorGUILayout.PropertyField(placement.FindPropertyRelative("slopeDegreesRange"));
            BiomeObjectPlacementEditorUtility.DrawObjectRules(
                placement.FindPropertyRelative("objectRules"),
                objectRulesTitle,
                defaultCategory,
                helpText);
        }

        private static string ResolveMacroName(BiomePalette palette, int macroClassIndex)
        {
            MacroBiomeSettings macro = palette.GetMacroBiomeSettings(macroClassIndex);
            return macro != null ? macro.macroName : $"Macro {macroClassIndex}";
        }

        private static void DrawValidation(BiomePalette palette)
        {
            if (palette == null)
            {
                return;
            }

            palette.EnsureMacroBiomes();
            DrawMacroSummary(palette);

            int issueCount = 0;
            if (palette.biomes == null || palette.biomes.Length == 0)
            {
                EditorGUILayout.HelpBox("No microbiomes are assigned. Painted colors will fall back to the fallback macro class.", MessageType.Warning);
                return;
            }

            for (int index = 0; index < palette.biomes.Length; index++)
            {
                BiomeDefinition biome = palette.biomes[index];
                if (biome == null)
                {
                    EditorGUILayout.HelpBox($"Microbiome slot {index} is empty.", MessageType.Warning);
                    issueCount++;
                    continue;
                }

                if (biome.macroClassIndex < 0 || biome.macroClassIndex >= palette.macroClassCount)
                {
                    EditorGUILayout.HelpBox($"{biome.name} has macro class {biome.macroClassIndex}, but valid classes are 0..{palette.macroClassCount - 1}.", MessageType.Warning);
                    issueCount++;
                }

                for (int otherIndex = index + 1; otherIndex < palette.biomes.Length; otherIndex++)
                {
                    BiomeDefinition other = palette.biomes[otherIndex];
                    if (other == null)
                    {
                        continue;
                    }

                    if (other.microBiomeId == biome.microBiomeId)
                    {
                        EditorGUILayout.HelpBox($"{biome.name} and {other.name} share micro biome id {biome.microBiomeId}.", MessageType.Warning);
                        issueCount++;
                    }

                    if (other.inputColor.Equals(biome.inputColor))
                    {
                        EditorGUILayout.HelpBox($"{biome.name} and {other.name} share input color {biome.inputColor}.", MessageType.Warning);
                        issueCount++;
                    }
                }
            }

            if (issueCount == 0)
            {
                EditorGUILayout.HelpBox("Palette validation passed.", MessageType.Info);
            }
        }

        private static void DrawMacroSummary(BiomePalette palette)
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Macro Groups", EditorStyles.boldLabel);
            for (int macroClass = 0; macroClass < Mathf.Max(1, palette.macroClassCount); macroClass++)
            {
                int count = 0;
                if (palette.biomes != null)
                {
                    foreach (BiomeDefinition biome in palette.biomes)
                    {
                        if (biome != null && Mathf.Clamp(biome.macroClassIndex, 0, palette.macroClassCount - 1) == macroClass)
                        {
                            count++;
                        }
                    }
                }

                MacroBiomeSettings settings = palette.GetMacroBiomeSettings(macroClass);
                string name = settings != null ? settings.macroName : $"Macro {macroClass}";
                EditorGUILayout.LabelField($"{macroClass}: {name}", $"{count} microbiome(s)");
            }
        }

        private static string ResolvePaletteFolder(BiomePalette palette)
        {
            const string fallback = "Assets/TerrainGAN/MicroBiomes";
            if (palette == null)
            {
                EnsureFolder(fallback);
                return fallback;
            }

            string path = AssetDatabase.GetAssetPath(palette);
            string folder = string.IsNullOrEmpty(path) ? fallback : Path.GetDirectoryName(path)?.Replace("\\", "/");
            if (string.IsNullOrEmpty(folder))
            {
                folder = fallback;
            }

            EnsureFolder(folder);
            return folder;
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
                string next = $"{current}/{parts[index]}";
                if (!AssetDatabase.IsValidFolder(next))
                {
                    AssetDatabase.CreateFolder(current, parts[index]);
                }

                current = next;
            }
        }

        private static int FindNextMicroBiomeId(BiomePalette palette)
        {
            int next = 0;
            if (palette?.biomes == null)
            {
                return next;
            }

            for (int index = 0; index < palette.biomes.Length; index++)
            {
                if (palette.biomes[index] != null)
                {
                    next = Mathf.Max(next, palette.biomes[index].microBiomeId + 1);
                }
            }

            return next;
        }

        private static Color32 NextColorForId(int id)
        {
            float hue = Mathf.Repeat(id * 0.173f, 1f);
            Color color = Color.HSVToRGB(hue, 0.55f, 0.85f);
            return color;
        }
    }
}
#endif
