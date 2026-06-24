#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Biomes;

namespace UnityTerrainGANerator.Editor
{
    public static class BiomeObjectPlacementEditorUtility
    {
        private const float DropZoneHeight = 42f;

        public static void DrawPrefabArrayDropZone(SerializedProperty arrayProperty, string title, string helpText)
        {
            if (arrayProperty == null)
            {
                return;
            }

            EditorGUILayout.LabelField(title, EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(arrayProperty, true);
            DrawDropZone(
                $"Drop prefab assets here for {title}",
                helpText,
                prefabs => AddPrefabsToArray(arrayProperty, prefabs));
        }

        public static void DrawObjectRules(
            SerializedProperty rulesProperty,
            string title,
            BiomeObjectCategory defaultCategory,
            string helpText = null)
        {
            if (rulesProperty == null)
            {
                return;
            }

            EditorGUILayout.LabelField(title, EditorStyles.boldLabel);
            if (!string.IsNullOrWhiteSpace(helpText))
            {
                EditorGUILayout.HelpBox(helpText, MessageType.None);
            }

            EditorGUILayout.PropertyField(rulesProperty, true);
            DrawDropZone(
                $"Drop prefab assets here to create {title}",
                "Each dropped prefab becomes one placement rule with editable density, height, slope and scale filters.",
                prefabs => AddPrefabsToRules(rulesProperty, prefabs, defaultCategory));
        }

        private static void DrawDropZone(string label, string helpText, System.Action<GameObject[]> onDrop)
        {
            if (!string.IsNullOrWhiteSpace(helpText))
            {
                EditorGUILayout.HelpBox(helpText, MessageType.None);
            }

            Rect rect = GUILayoutUtility.GetRect(0f, DropZoneHeight, GUILayout.ExpandWidth(true));
            GUI.Box(rect, label, EditorStyles.helpBox);

            Event current = Event.current;
            if (!rect.Contains(current.mousePosition))
            {
                return;
            }

            if (current.type == EventType.DragUpdated || current.type == EventType.DragPerform)
            {
                GameObject[] prefabs = ResolveDraggedPrefabs();
                DragAndDrop.visualMode = prefabs.Length > 0 ? DragAndDropVisualMode.Copy : DragAndDropVisualMode.Rejected;

                if (current.type == EventType.DragPerform && prefabs.Length > 0)
                {
                    DragAndDrop.AcceptDrag();
                    onDrop(prefabs);
                    GUI.changed = true;
                }

                current.Use();
            }
        }

        private static GameObject[] ResolveDraggedPrefabs()
        {
            Object[] references = DragAndDrop.objectReferences;
            if (references == null || references.Length == 0)
            {
                return new GameObject[0];
            }

            System.Collections.Generic.List<GameObject> prefabs = new System.Collections.Generic.List<GameObject>();
            foreach (Object reference in references)
            {
                GameObject prefab = ResolvePrefab(reference);
                if (prefab != null && !prefabs.Contains(prefab))
                {
                    prefabs.Add(prefab);
                }
            }

            return prefabs.ToArray();
        }

        private static GameObject ResolvePrefab(Object reference)
        {
            GameObject gameObject = reference as GameObject;
            if (gameObject == null)
            {
                return null;
            }

            if (EditorUtility.IsPersistent(gameObject))
            {
                return gameObject;
            }

            GameObject source = PrefabUtility.GetCorrespondingObjectFromSource(gameObject);
            return source != null && EditorUtility.IsPersistent(source) ? source : null;
        }

        private static void AddPrefabsToArray(SerializedProperty arrayProperty, GameObject[] prefabs)
        {
            if (prefabs == null || prefabs.Length == 0)
            {
                return;
            }

            Undo.RecordObject(arrayProperty.serializedObject.targetObject, "Add biome prefab assets");
            foreach (GameObject prefab in prefabs)
            {
                if (ContainsPrefab(arrayProperty, prefab))
                {
                    continue;
                }

                int index = arrayProperty.arraySize;
                arrayProperty.InsertArrayElementAtIndex(index);
                arrayProperty.GetArrayElementAtIndex(index).objectReferenceValue = prefab;
            }

            arrayProperty.serializedObject.ApplyModifiedProperties();
            EditorUtility.SetDirty(arrayProperty.serializedObject.targetObject);
        }

        private static void AddPrefabsToRules(
            SerializedProperty rulesProperty,
            GameObject[] prefabs,
            BiomeObjectCategory defaultCategory)
        {
            if (prefabs == null || prefabs.Length == 0)
            {
                return;
            }

            Undo.RecordObject(rulesProperty.serializedObject.targetObject, "Add biome object placement rules");
            foreach (GameObject prefab in prefabs)
            {
                if (ContainsRulePrefab(rulesProperty, prefab))
                {
                    continue;
                }

                int index = rulesProperty.arraySize;
                rulesProperty.InsertArrayElementAtIndex(index);
                SerializedProperty rule = rulesProperty.GetArrayElementAtIndex(index);
                rule.FindPropertyRelative("ruleName").stringValue = ObjectNames.NicifyVariableName(prefab.name);
                rule.FindPropertyRelative("enabled").boolValue = true;
                rule.FindPropertyRelative("category").enumValueIndex = (int)defaultCategory;
                rule.FindPropertyRelative("prefab").objectReferenceValue = prefab;
                rule.FindPropertyRelative("densityPerSquareKm").floatValue = DefaultDensity(defaultCategory);
                rule.FindPropertyRelative("normalizedHeightRange").vector2Value = new Vector2(0f, 1f);
                rule.FindPropertyRelative("slopeDegreesRange").vector2Value = DefaultSlope(defaultCategory);
                rule.FindPropertyRelative("uniformScaleRange").vector2Value = DefaultScale(defaultCategory);
                rule.FindPropertyRelative("minSpacingMeters").floatValue = DefaultSpacing(defaultCategory);
                rule.FindPropertyRelative("noiseScale").floatValue = 1f;
                rule.FindPropertyRelative("noiseInfluence").floatValue = 0.35f;
                rule.FindPropertyRelative("verticalOffsetMeters").vector2Value = Vector2.zero;
                rule.FindPropertyRelative("alignToTerrainNormal").boolValue = defaultCategory != BiomeObjectCategory.Tree;
                rule.FindPropertyRelative("normalAlignment").floatValue = defaultCategory == BiomeObjectCategory.Tree ? 0.35f : 0.85f;
                rule.FindPropertyRelative("randomYaw").floatValue = 1f;
                rule.FindPropertyRelative("placementJitter").floatValue = 0.35f;
            }

            rulesProperty.serializedObject.ApplyModifiedProperties();
            EditorUtility.SetDirty(rulesProperty.serializedObject.targetObject);
        }

        private static bool ContainsPrefab(SerializedProperty arrayProperty, GameObject prefab)
        {
            for (int index = 0; index < arrayProperty.arraySize; index++)
            {
                if (arrayProperty.GetArrayElementAtIndex(index).objectReferenceValue == prefab)
                {
                    return true;
                }
            }

            return false;
        }

        private static bool ContainsRulePrefab(SerializedProperty rulesProperty, GameObject prefab)
        {
            for (int index = 0; index < rulesProperty.arraySize; index++)
            {
                SerializedProperty element = rulesProperty.GetArrayElementAtIndex(index);
                if (element.FindPropertyRelative("prefab").objectReferenceValue == prefab)
                {
                    return true;
                }
            }

            return false;
        }

        private static float DefaultDensity(BiomeObjectCategory category)
        {
            switch (category)
            {
                case BiomeObjectCategory.Tree:
                    return 120f;
                case BiomeObjectCategory.Detail:
                    return 300f;
                case BiomeObjectCategory.Rock:
                    return 45f;
                default:
                    return 25f;
            }
        }

        private static Vector2 DefaultSlope(BiomeObjectCategory category)
        {
            switch (category)
            {
                case BiomeObjectCategory.Tree:
                    return new Vector2(0f, 35f);
                case BiomeObjectCategory.Rock:
                    return new Vector2(5f, 65f);
                default:
                    return new Vector2(0f, 45f);
            }
        }

        private static Vector2 DefaultScale(BiomeObjectCategory category)
        {
            switch (category)
            {
                case BiomeObjectCategory.Tree:
                    return new Vector2(0.85f, 1.35f);
                case BiomeObjectCategory.Detail:
                    return new Vector2(0.7f, 1.2f);
                case BiomeObjectCategory.Rock:
                    return new Vector2(0.6f, 1.6f);
                default:
                    return new Vector2(0.8f, 1.2f);
            }
        }

        private static float DefaultSpacing(BiomeObjectCategory category)
        {
            switch (category)
            {
                case BiomeObjectCategory.Tree:
                    return 5f;
                case BiomeObjectCategory.Rock:
                    return 3f;
                case BiomeObjectCategory.Detail:
                    return 0.75f;
                default:
                    return 1f;
            }
        }
    }
}
#endif
