using UnityEditor;
using UnityEngine;

namespace UnityTerrainGANerator.Editor
{
    public static class TerrainGANTextureImportUtility
    {
        public static void MakeSelectedTextureReadable()
        {
            Texture2D texture = Selection.activeObject as Texture2D;
            if (texture == null)
            {
                Debug.LogWarning("Select a Texture2D asset first.");
                return;
            }

            EnsureReadable(texture, true);
        }

        public static bool EnsureReadable(Texture2D texture, bool logResult)
        {
            if (texture == null)
            {
                return false;
            }

            string path = AssetDatabase.GetAssetPath(texture);
            TextureImporter importer = AssetImporter.GetAtPath(path) as TextureImporter;
            if (importer == null)
            {
                if (logResult && !texture.isReadable)
                {
                    Debug.LogWarning($"Texture is not an imported asset and cannot be made readable automatically: {texture.name}");
                }

                return texture.isReadable;
            }

            bool changed = false;
            if (!importer.isReadable)
            {
                importer.isReadable = true;
                changed = true;
            }

            if (importer.textureCompression != TextureImporterCompression.Uncompressed)
            {
                importer.textureCompression = TextureImporterCompression.Uncompressed;
                changed = true;
            }

            if (changed)
            {
                importer.SaveAndReimport();
                AssetDatabase.Refresh();
                if (logResult)
                {
                    Debug.Log($"Texture is now readable: {path}");
                }
            }
            else if (logResult)
            {
                Debug.Log($"Texture is already readable: {path}");
            }

            return true;
        }
    }
}
