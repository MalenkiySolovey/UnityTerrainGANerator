#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Biomes;

namespace UnityTerrainGANerator.Editor
{
    public sealed class TerrainGANBiomePaletteWindow : EditorWindow
    {
        private const string DefaultPalettePath = "Assets/TerrainGAN/Palettes/Dataset_BiomePalette.asset";

        private BiomePalette palette;
        private UnityEditor.Editor paletteEditor;
        private Vector2 scrollPosition;

        [MenuItem("Tools/Terrain GAN/Biome Palette Manager")]
        public static void Open()
        {
            TerrainGANBiomePaletteWindow window = GetWindow<TerrainGANBiomePaletteWindow>("Terrain GAN Biomes");
            window.minSize = new Vector2(460f, 620f);
            window.LoadDefaults();
        }

        [MenuItem("Tools/Terrain GAN/Create/Biome Palette")]
        public static void CreateBiomePaletteAsset()
        {
            string path = EditorUtility.SaveFilePanelInProject(
                "Create Biome Palette",
                "Biome Palette.asset",
                "asset",
                "Choose where to save the biome palette.",
                "Assets/TerrainGAN/Palettes");

            if (string.IsNullOrWhiteSpace(path))
            {
                return;
            }

            BiomePalette asset = ScriptableObject.CreateInstance<BiomePalette>();
            asset.EnsureMacroBiomes();
            AssetDatabase.CreateAsset(asset, path);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Selection.activeObject = asset;
        }

        [MenuItem("Tools/Terrain GAN/Create/Microbiome")]
        public static void CreateMicrobiomeAsset()
        {
            BiomePalette selectedPalette = Selection.activeObject as BiomePalette;
            BiomePaletteEditor.CreateMicrobiomeAsset(selectedPalette);
        }

        private void OnEnable()
        {
            LoadDefaults();
        }

        private void OnGUI()
        {
            scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);
            EditorGUILayout.LabelField("Biome Palette Manager", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox("Create macrobiomes for the model contract and microbiomes for user painting/gameplay. Microbiomes map back to macro classes before ONNX inference.", MessageType.Info);

            EditorGUI.BeginChangeCheck();
            palette = (BiomePalette)EditorGUILayout.ObjectField("Biome Palette", palette, typeof(BiomePalette), false);
            if (EditorGUI.EndChangeCheck())
            {
                paletteEditor = null;
            }

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Create Palette"))
            {
                CreateBiomePaletteAsset();
            }

            using (new EditorGUI.DisabledScope(palette == null))
            {
                if (GUILayout.Button("Create Microbiome"))
                {
                    BiomePaletteEditor.CreateMicrobiomeAsset(palette);
                }
            }
            EditorGUILayout.EndHorizontal();

            if (palette != null)
            {
                UnityEditor.Editor.CreateCachedEditor(palette, typeof(BiomePaletteEditor), ref paletteEditor);
                paletteEditor.OnInspectorGUI();
            }

            EditorGUILayout.EndScrollView();
        }

        private void LoadDefaults()
        {
            if (palette == null)
            {
                palette = AssetDatabase.LoadAssetAtPath<BiomePalette>(DefaultPalettePath);
            }
        }
    }
}
#endif
