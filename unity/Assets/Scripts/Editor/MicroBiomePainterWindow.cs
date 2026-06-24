#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Biomes;
using UnityTerrainGANerator.Preview;

namespace UnityTerrainGANerator.Editor
{
    public sealed class MicroBiomePainterWindow : EditorWindow
    {
        private const string DefaultProfilePath = "Assets/TerrainGAN/Profiles/DemoPreviewProfile.asset";

        private TerrainGANPreviewProfile profile;
        private BiomePalette palette;
        private Texture2D biomeTexture;
        private BiomeDefinition activeBiome;
        private Vector2 scroll;
        private int textureSize = 256;
        private int brushRadius = 8;
        private bool unsavedTextureChanges;

        [MenuItem("Tools/Terrain GAN/Micro Biome Painter")]
        public static void Open()
        {
            MicroBiomePainterWindow window = GetWindow<MicroBiomePainterWindow>("Micro Biome Painter");
            window.minSize = new Vector2(460f, 620f);
            window.LoadDefaults();
        }

        private void OnEnable()
        {
            LoadDefaults();
        }

        private void OnGUI()
        {
            scroll = EditorGUILayout.BeginScrollView(scroll);
            EditorGUILayout.LabelField("Micro Biome Painter", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            DrawProfileSettings();
            DrawTextureTools();
            DrawBiomeSelector();
            DrawCanvas();
            DrawStatus();

            EditorGUILayout.EndScrollView();
        }

        private void DrawProfileSettings()
        {
            EditorGUI.BeginChangeCheck();
            profile = (TerrainGANPreviewProfile)EditorGUILayout.ObjectField("Preview Profile", profile, typeof(TerrainGANPreviewProfile), false);
            palette = (BiomePalette)EditorGUILayout.ObjectField("Biome Palette", palette, typeof(BiomePalette), false);
            biomeTexture = (Texture2D)EditorGUILayout.ObjectField("Biome Texture", biomeTexture, typeof(Texture2D), false);
            if (EditorGUI.EndChangeCheck())
            {
                if (profile != null)
                {
                    profile.palette = palette;
                    profile.biomeTexture = biomeTexture;
                    EditorUtility.SetDirty(profile);
                }

                SelectDefaultBiomeIfNeeded();
            }

            using (new EditorGUI.DisabledScope(profile == null))
            {
                if (GUILayout.Button("Use Profile Inputs"))
                {
                    palette = profile.palette;
                    biomeTexture = profile.biomeTexture;
                    SelectDefaultBiomeIfNeeded();
                }
            }
        }

        private void DrawTextureTools()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Texture", EditorStyles.boldLabel);
            textureSize = Mathf.Clamp(EditorGUILayout.IntField("New Texture Size", textureSize), 16, 4096);
            brushRadius = Mathf.Clamp(EditorGUILayout.IntField("Brush Radius", brushRadius), 1, 256);

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("New Texture Asset"))
            {
                CreateTextureAsset();
            }

            using (new EditorGUI.DisabledScope(biomeTexture == null || activeBiome == null))
            {
                if (GUILayout.Button("Fill With Active"))
                {
                    FillTexture();
                }
            }

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            using (new EditorGUI.DisabledScope(biomeTexture == null))
            {
                if (GUILayout.Button("Save Texture"))
                {
                    SaveTexture();
                }
            }

            using (new EditorGUI.DisabledScope(biomeTexture == null || palette == null))
            {
                if (GUILayout.Button("Export Macro Preview"))
                {
                    TerrainGANPreviewExportUtility.ExportMacroBiomeTexture(biomeTexture, palette);
                }
            }

            EditorGUILayout.EndHorizontal();
        }

        private void DrawBiomeSelector()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Active Micro Biome", EditorStyles.boldLabel);
            activeBiome = (BiomeDefinition)EditorGUILayout.ObjectField("Biome", activeBiome, typeof(BiomeDefinition), false);

            if (palette == null || palette.biomes == null)
            {
                return;
            }

            int macroClassCount = Mathf.Max(1, palette.macroClassCount);
            for (int macroClass = 0; macroClass < macroClassCount; macroClass++)
            {
                EditorGUILayout.LabelField($"Macro {macroClass}", EditorStyles.miniBoldLabel);
                EditorGUILayout.BeginHorizontal();
                int column = 0;
                foreach (BiomeDefinition biome in palette.biomes)
                {
                    if (biome == null || Mathf.Clamp(biome.macroClassIndex, 0, macroClassCount - 1) != macroClass)
                    {
                        continue;
                    }

                    Color previousColor = GUI.backgroundColor;
                    GUI.backgroundColor = ToGuiColor(biome.inputColor);
                    string label = string.IsNullOrWhiteSpace(biome.biomeName) ? biome.name : biome.biomeName;
                    if (GUILayout.Button(label, GUILayout.MaxWidth(120f)))
                    {
                        activeBiome = biome;
                        GUI.FocusControl(null);
                    }

                    GUI.backgroundColor = previousColor;
                    column++;
                    if (column % 3 == 0)
                    {
                        EditorGUILayout.EndHorizontal();
                        EditorGUILayout.BeginHorizontal();
                    }
                }

                EditorGUILayout.EndHorizontal();
            }
        }

        private void DrawCanvas()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Paint", EditorStyles.boldLabel);
            if (biomeTexture == null)
            {
                EditorGUILayout.HelpBox("Create or assign a biome texture first.", MessageType.Info);
                return;
            }

            float size = Mathf.Min(position.width - 42f, 512f);
            Rect rect = GUILayoutUtility.GetRect(size, size, GUILayout.ExpandWidth(false));
            GUI.DrawTexture(rect, biomeTexture, ScaleMode.StretchToFill, false);
            Handles.DrawSolidRectangleWithOutline(rect, Color.clear, new Color(0f, 0f, 0f, 0.45f));
            HandlePaintEvent(rect);
        }

        private void HandlePaintEvent(Rect rect)
        {
            Event current = Event.current;
            if (activeBiome == null || biomeTexture == null)
            {
                return;
            }

            bool wantsPaint = current.type == EventType.MouseDown || current.type == EventType.MouseDrag;
            if (!wantsPaint || current.button != 0 || !rect.Contains(current.mousePosition))
            {
                return;
            }

            TerrainGANTextureImportUtility.EnsureReadable(biomeTexture, false);
            Undo.RegisterCompleteObjectUndo(biomeTexture, "Paint Micro Biome");
            float u = Mathf.InverseLerp(rect.xMin, rect.xMax, current.mousePosition.x);
            float vFromTop = Mathf.InverseLerp(rect.yMin, rect.yMax, current.mousePosition.y);
            int x = Mathf.Clamp(Mathf.RoundToInt(u * (biomeTexture.width - 1)), 0, biomeTexture.width - 1);
            int y = Mathf.Clamp(Mathf.RoundToInt((1f - vFromTop) * (biomeTexture.height - 1)), 0, biomeTexture.height - 1);
            MicroBiomeTexturePainter.PaintCircle(biomeTexture, x, y, brushRadius, activeBiome.inputColor, true);
            EditorUtility.SetDirty(biomeTexture);
            unsavedTextureChanges = true;
            current.Use();
            Repaint();
        }

        private void DrawStatus()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            if (palette == null)
            {
                EditorGUILayout.HelpBox("Biome Palette is not assigned.", MessageType.Warning);
            }

            if (activeBiome == null)
            {
                EditorGUILayout.HelpBox("Select an active micro biome before painting.", MessageType.Info);
            }

            if (unsavedTextureChanges)
            {
                EditorGUILayout.HelpBox("Texture has unsaved paint changes.", MessageType.Info);
            }
        }

        private void CreateTextureAsset()
        {
            if (palette == null)
            {
                Debug.LogWarning("Assign a Biome Palette before creating a texture.");
                return;
            }

            SelectDefaultBiomeIfNeeded();
            string path = EditorUtility.SaveFilePanelInProject(
                "Create Micro Biome Texture",
                "micro_biome_map",
                "asset",
                "Choose where to save the editable micro biome texture."
            );

            if (string.IsNullOrEmpty(path))
            {
                return;
            }

            Texture2D texture = new Texture2D(textureSize, textureSize, TextureFormat.RGBA32, false)
            {
                name = Path.GetFileNameWithoutExtension(path),
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            MicroBiomeTexturePainter.FillTexture(texture, activeBiome != null ? activeBiome.inputColor : palette.FindMacroColor(palette.fallbackMacroClassIndex));
            AssetDatabase.CreateAsset(texture, path);
            AssetDatabase.SaveAssets();
            biomeTexture = texture;
            AssignTextureToProfile();
            unsavedTextureChanges = false;
        }

        private void FillTexture()
        {
            TerrainGANTextureImportUtility.EnsureReadable(biomeTexture, false);
            Undo.RegisterCompleteObjectUndo(biomeTexture, "Fill Micro Biome Texture");
            MicroBiomeTexturePainter.FillTexture(biomeTexture, activeBiome.inputColor);
            EditorUtility.SetDirty(biomeTexture);
            unsavedTextureChanges = true;
        }

        private void SaveTexture()
        {
            if (biomeTexture == null)
            {
                return;
            }

            string path = AssetDatabase.GetAssetPath(biomeTexture);
            if (string.IsNullOrEmpty(path))
            {
                Debug.LogWarning("Texture is not an asset. Create a texture asset first.");
                return;
            }

            string extension = Path.GetExtension(path).ToLowerInvariant();
            if (extension == ".png")
            {
                File.WriteAllBytes(path, biomeTexture.EncodeToPNG());
                AssetDatabase.ImportAsset(path);
                biomeTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
                TerrainGANTextureImportUtility.EnsureReadable(biomeTexture, false);
                AssignTextureToProfile();
            }
            else
            {
                EditorUtility.SetDirty(biomeTexture);
                AssetDatabase.SaveAssets();
            }

            unsavedTextureChanges = false;
        }

        private void LoadDefaults()
        {
            profile = AssetDatabase.LoadAssetAtPath<TerrainGANPreviewProfile>(DefaultProfilePath);
            if (profile != null)
            {
                palette = profile.palette;
                biomeTexture = profile.biomeTexture;
            }

            SelectDefaultBiomeIfNeeded();
        }

        private void AssignTextureToProfile()
        {
            if (profile == null)
            {
                return;
            }

            profile.palette = palette;
            profile.biomeTexture = biomeTexture;
            EditorUtility.SetDirty(profile);
        }

        private void SelectDefaultBiomeIfNeeded()
        {
            if (activeBiome != null || palette == null)
            {
                return;
            }

            activeBiome = palette.FindFallbackBiome();
            if (activeBiome == null && palette.biomes != null && palette.biomes.Length > 0)
            {
                activeBiome = palette.biomes[0];
            }
        }

        private static Color ToGuiColor(Color32 color)
        {
            return new Color(color.r / 255f, color.g / 255f, color.b / 255f, 1f);
        }
    }
}
#endif
