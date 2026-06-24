using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Inference;

namespace UnityTerrainGANerator.Editor
{
    [InitializeOnLoad]
    internal static class TerrainGANInferenceCleanup
    {
        static TerrainGANInferenceCleanup()
        {
            AssemblyReloadEvents.beforeAssemblyReload += DisposeAllInferenceControllers;
            EditorApplication.quitting += DisposeAllInferenceControllers;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingEditMode ||
                state == PlayModeStateChange.ExitingPlayMode)
            {
                DisposeAllInferenceControllers();
            }
        }

        private static void DisposeAllInferenceControllers()
        {
            ONNXInferenceController[] controllers = Resources.FindObjectsOfTypeAll<ONNXInferenceController>();
            foreach (ONNXInferenceController controller in controllers)
            {
                if (controller != null)
                {
                    controller.Dispose();
                }
            }
        }
    }
}
