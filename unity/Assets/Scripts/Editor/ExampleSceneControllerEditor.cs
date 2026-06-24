using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Demo;

namespace UnityTerrainGANerator.Editor
{
    [CustomEditor(typeof(ExampleSceneController))]
    public sealed class ExampleSceneControllerEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            EditorGUILayout.Space();

            if (GUILayout.Button("Apply Preview Profile"))
            {
                foreach (Object selectedTarget in targets)
                {
                    ExampleSceneController controller = (ExampleSceneController)selectedTarget;
                    controller.ApplyPreviewProfile();
                    EditorUtility.SetDirty(controller);
                }
            }

            if (GUILayout.Button("Generate Default Terrain"))
            {
                foreach (Object selectedTarget in targets)
                {
                    ExampleSceneController controller = (ExampleSceneController)selectedTarget;
                    Texture2D texture = controller.previewProfile != null && controller.previewProfile.biomeTexture != null
                        ? controller.previewProfile.biomeTexture
                        : controller.defaultBiomeTexture;
                    TerrainGANTextureImportUtility.EnsureReadable(texture, true);
                    controller.GenerateDefault();
                    if (controller.terrainGenerator != null &&
                        controller.terrainGenerator.terrainBuilder != null &&
                        controller.terrainGenerator.terrainBuilder.targetTerrain != null)
                    {
                        EditorUtility.SetDirty(controller.terrainGenerator.terrainBuilder.targetTerrain.terrainData);
                    }
                }
            }

            if (GUILayout.Button("Export Heightmap PNG"))
            {
                foreach (Object selectedTarget in targets)
                {
                    ExampleSceneController controller = (ExampleSceneController)selectedTarget;
                    if (controller.terrainGenerator != null &&
                        controller.terrainGenerator.terrainBuilder != null)
                    {
                        TerrainGANPreviewExportUtility.ExportHeightmapPng(controller.terrainGenerator.terrainBuilder.targetTerrain);
                    }
                }
            }
        }
    }
}
