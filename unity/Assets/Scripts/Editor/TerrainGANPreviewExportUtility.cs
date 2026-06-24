using System;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityTerrainGANerator.Biomes;

namespace UnityTerrainGANerator.Editor
{
    public static class TerrainGANPreviewExportUtility
    {
        private const string ExportFolder = "Assets/TerrainGAN/Exports";

        public static string ExportHeightmapPng(Terrain terrain)
        {
            if (terrain == null || terrain.terrainData == null)
            {
                Debug.LogError("Heightmap export failed: Terrain is not assigned.");
                return null;
            }

            Directory.CreateDirectory(ExportFolder);

            TerrainData data = terrain.terrainData;
            int resolution = data.heightmapResolution;
            float[,] heights = data.GetHeights(0, 0, resolution, resolution);
            Texture2D texture = new Texture2D(resolution, resolution, TextureFormat.RGBA32, false);
            Color32[] pixels = new Color32[resolution * resolution];

            for (int y = 0; y < resolution; y++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    byte value = (byte)Mathf.RoundToInt(Mathf.Clamp01(heights[y, x]) * 255f);
                    int flippedY = resolution - 1 - y;
                    pixels[flippedY * resolution + x] = new Color32(value, value, value, 255);
                }
            }

            texture.SetPixels32(pixels);
            texture.Apply();

            string path = $"{ExportFolder}/heightmap_{DateTime.Now:yyyyMMdd_HHmmss}.png";
            File.WriteAllBytes(path, texture.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(texture);
            AssetDatabase.ImportAsset(path);
            Debug.Log($"Terrain GAN heightmap exported: {path}");
            return path;
        }

        public static string ExportBiomeTexture(Texture2D biomeTexture)
        {
            if (biomeTexture == null)
            {
                Debug.LogError("Biome texture export failed: texture is not assigned.");
                return null;
            }

            TerrainGANTextureImportUtility.EnsureReadable(biomeTexture, false);
            Directory.CreateDirectory(ExportFolder);
            string sourcePath = AssetDatabase.GetAssetPath(biomeTexture);
            string path = $"{ExportFolder}/biome_map_{DateTime.Now:yyyyMMdd_HHmmss}.png";

            if (!string.IsNullOrEmpty(sourcePath) && File.Exists(sourcePath))
            {
                File.Copy(sourcePath, path, true);
            }
            else
            {
                File.WriteAllBytes(path, biomeTexture.EncodeToPNG());
            }

            AssetDatabase.ImportAsset(path);
            Debug.Log($"Terrain GAN biome texture exported: {path}");
            return path;
        }

        public static string ExportGeneratedTexture(Texture2D generatedTexture)
        {
            if (generatedTexture == null)
            {
                Debug.LogError("Generated texture export failed: generate terrain with Generated ONNX Texture mode first.");
                return null;
            }

            Directory.CreateDirectory(ExportFolder);
            string path = $"{ExportFolder}/generated_texture_{DateTime.Now:yyyyMMdd_HHmmss}.png";
            File.WriteAllBytes(path, generatedTexture.EncodeToPNG());
            AssetDatabase.ImportAsset(path);
            Debug.Log($"Terrain GAN generated texture exported: {path}");
            return path;
        }

        public static string ExportMacroBiomeTexture(Texture2D biomeTexture, BiomePalette palette)
        {
            if (biomeTexture == null)
            {
                Debug.LogError("Macro biome texture export failed: texture is not assigned.");
                return null;
            }

            if (palette == null)
            {
                Debug.LogError("Macro biome texture export failed: palette is not assigned.");
                return null;
            }

            TerrainGANTextureImportUtility.EnsureReadable(biomeTexture, false);
            Directory.CreateDirectory(ExportFolder);
            Color32[] sourcePixels = biomeTexture.GetPixels32();
            Color32[] macroPixels = new Color32[sourcePixels.Length];

            for (int i = 0; i < sourcePixels.Length; i++)
            {
                macroPixels[i] = palette.ConvertToMacroColor(sourcePixels[i]);
            }

            Texture2D texture = new Texture2D(biomeTexture.width, biomeTexture.height, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            texture.SetPixels32(macroPixels);
            texture.Apply(false, false);

            string path = $"{ExportFolder}/macro_biome_map_{DateTime.Now:yyyyMMdd_HHmmss}.png";
            File.WriteAllBytes(path, texture.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(texture);
            AssetDatabase.ImportAsset(path);
            Debug.Log($"Terrain GAN macro biome texture exported: {path}");
            return path;
        }
    }
}
