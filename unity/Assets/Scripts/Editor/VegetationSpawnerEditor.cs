#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.TerrainGeneration;

namespace UnityTerrainGANerator.Editor
{
    [CustomEditor(typeof(VegetationSpawner))]
    public sealed class VegetationSpawnerEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "Object Rules place prefab instances inside macro/microbiome areas using density per square kilometer, terrain height, surface slope, jitter, noise, spacing, scale and normal alignment filters.",
                MessageType.Info);

            if (GUILayout.Button("Respawn Objects From Last Terrain GAN Map"))
            {
                foreach (Object selectedTarget in targets)
                {
                    VegetationSpawner spawner = (VegetationSpawner)selectedTarget;
                    TerrainGenerator generator = spawner.GetComponent<TerrainGenerator>();
                    if (generator == null)
                    {
                        Debug.LogWarning("No TerrainGenerator found on this GameObject. Generate terrain first from TerrainGANRuntime.");
                        continue;
                    }

                    generator.RespawnVegetation();
                    MarkTerrainDirty(spawner);
                }
            }

            if (GUILayout.Button("Clear Spawned Objects"))
            {
                foreach (Object selectedTarget in targets)
                {
                    VegetationSpawner spawner = (VegetationSpawner)selectedTarget;
                    spawner.ClearSpawnedObjects();
                    MarkTerrainDirty(spawner);
                }
            }
        }

        private static void MarkTerrainDirty(VegetationSpawner spawner)
        {
            if (spawner == null)
            {
                return;
            }

            EditorUtility.SetDirty(spawner);
            if (spawner.targetTerrain != null)
            {
                EditorUtility.SetDirty(spawner.targetTerrain);
                if (spawner.targetTerrain.terrainData != null)
                {
                    EditorUtility.SetDirty(spawner.targetTerrain.terrainData);
                }
            }
        }
    }
}
#endif
