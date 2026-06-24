#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Biomes;

namespace UnityTerrainGANerator.Editor
{
    [CustomEditor(typeof(BiomeDefinition))]
    public sealed class BiomeDefinitionEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.LabelField("Microbiome Identity", EditorStyles.boldLabel);
            Draw("biomeName");
            Draw("microBiomeId");
            Draw("macroClassIndex");
            Draw("inputColor");
            Draw("macroInputColor");
            EditorGUILayout.HelpBox("The painter can use unique microbiome colors. Before ONNX inference, every microbiome is converted to its macro class channel.", MessageType.Info);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Terrain Paint", EditorStyles.boldLabel);
            Draw("terrainLayers", true);
            Draw("terrainLayerWeights", true);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Object Placement Shortcuts", EditorStyles.boldLabel);
            BiomeObjectPlacementEditorUtility.DrawPrefabArrayDropZone(
                serializedObject.FindProperty("treePrefabs"),
                "Tree Prefabs",
                "Used by the current VegetationSpawner tree prototype pass.");
            BiomeObjectPlacementEditorUtility.DrawPrefabArrayDropZone(
                serializedObject.FindProperty("detailPrefabs"),
                "Detail / Plant Prefabs",
                "Used by the current VegetationSpawner detail prefab pass.");
            BiomeObjectPlacementEditorUtility.DrawPrefabArrayDropZone(
                serializedObject.FindProperty("rockPrefabs"),
                "Rock Prefabs",
                "Used by the current VegetationSpawner rock prefab pass.");
            Draw("heightRange");
            Draw("slopeRange");
            Draw("treeDensity");
            Draw("detailDensity");
            Draw("rockDensity");
            Draw("noiseScale");
            Draw("treeScaleRange");
            Draw("objectScaleRange");
            Draw("placementJitter");
            Draw("alignObjectsToTerrainNormal");
            Draw("normalAlignment");
            Draw("seedOffset");

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Detailed Placement Rules", EditorStyles.boldLabel);
            Draw("inheritMacroPlacement");
            SerializedProperty placement = serializedObject.FindProperty("placement");
            if (placement != null)
            {
                EditorGUILayout.PropertyField(placement.FindPropertyRelative("terrainLayers"), true);
                EditorGUILayout.PropertyField(placement.FindPropertyRelative("densityMultiplier"));
                EditorGUILayout.PropertyField(placement.FindPropertyRelative("normalizedHeightRange"));
                EditorGUILayout.PropertyField(placement.FindPropertyRelative("slopeDegreesRange"));
                BiomeObjectPlacementEditorUtility.DrawObjectRules(
                    placement.FindPropertyRelative("objectRules"),
                    "Microbiome Object Rules",
                    BiomeObjectCategory.Prop,
                    "Stored on this microbiome. These rules can describe trees, rocks, props and plants with their own filters.");
            }
            EditorGUILayout.HelpBox("Detailed rules are used by VegetationSpawner object-rule placement. Current terrain generation still sends only macro channels to the ONNX model.", MessageType.None);

            serializedObject.ApplyModifiedProperties();
        }

        private void Draw(string propertyName, bool includeChildren = false)
        {
            SerializedProperty property = serializedObject.FindProperty(propertyName);
            if (property != null)
            {
                EditorGUILayout.PropertyField(property, includeChildren);
            }
        }
    }
}
#endif
