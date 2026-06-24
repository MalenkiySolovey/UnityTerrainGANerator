using UnityEngine;

namespace UnityTerrainGANerator.Biomes
{
    public sealed class MicroBiomeTexturePainter : MonoBehaviour
    {
        public BiomePalette palette;
        public Texture2D biomeTexture;
        public BiomeDefinition activeBiome;
        public int textureSize = 256;
        public int brushRadiusPixels = 8;
        public bool applyAfterPaint = true;

        public Texture2D CreateTexture(BiomeDefinition fillBiome)
        {
            int size = Mathf.Max(1, textureSize);
            Texture2D texture = new Texture2D(size, size, TextureFormat.RGBA32, false)
            {
                name = "MicroBiomeMap",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };

            FillTexture(texture, ResolveColor(fillBiome));
            biomeTexture = texture;
            return texture;
        }

        public void Fill(BiomeDefinition biome)
        {
            EnsureTexture();
            FillTexture(biomeTexture, ResolveColor(biome));
        }

        public void PaintNormalized(Vector2 normalizedPosition, BiomeDefinition biome)
        {
            EnsureTexture();
            int x = Mathf.RoundToInt(Mathf.Clamp01(normalizedPosition.x) * (biomeTexture.width - 1));
            int y = Mathf.RoundToInt(Mathf.Clamp01(normalizedPosition.y) * (biomeTexture.height - 1));
            PaintPixel(x, y, Mathf.Max(1, brushRadiusPixels), biome);
        }

        public void PaintPixel(int centerX, int centerY, int radiusPixels, BiomeDefinition biome)
        {
            EnsureTexture();
            PaintCircle(biomeTexture, centerX, centerY, radiusPixels, ResolveColor(biome), applyAfterPaint);
        }

        public Texture2D BuildMacroTexture()
        {
            EnsureTexture();
            if (palette == null)
            {
                throw new System.InvalidOperationException("Biome palette is not assigned.");
            }

            Texture2D macroTexture = new Texture2D(biomeTexture.width, biomeTexture.height, TextureFormat.RGBA32, false)
            {
                name = biomeTexture.name + "_Macro",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            Color32[] sourcePixels = biomeTexture.GetPixels32();
            Color32[] macroPixels = new Color32[sourcePixels.Length];
            for (int i = 0; i < sourcePixels.Length; i++)
            {
                macroPixels[i] = palette.ConvertToMacroColor(sourcePixels[i]);
            }

            macroTexture.SetPixels32(macroPixels);
            macroTexture.Apply(false, false);
            return macroTexture;
        }

        public static void FillTexture(Texture2D texture, Color32 color)
        {
            if (texture == null)
            {
                throw new System.ArgumentNullException(nameof(texture));
            }

            Color32[] pixels = new Color32[texture.width * texture.height];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = color;
            }

            texture.SetPixels32(pixels);
            texture.Apply(false, false);
        }

        public static void PaintCircle(Texture2D texture, int centerX, int centerY, int radiusPixels, Color32 color, bool apply)
        {
            if (texture == null)
            {
                throw new System.ArgumentNullException(nameof(texture));
            }

            int radius = Mathf.Max(1, radiusPixels);
            int radiusSquared = radius * radius;
            int minX = Mathf.Max(0, centerX - radius);
            int maxX = Mathf.Min(texture.width - 1, centerX + radius);
            int minY = Mathf.Max(0, centerY - radius);
            int maxY = Mathf.Min(texture.height - 1, centerY + radius);

            for (int y = minY; y <= maxY; y++)
            {
                int dy = y - centerY;
                for (int x = minX; x <= maxX; x++)
                {
                    int dx = x - centerX;
                    if (dx * dx + dy * dy <= radiusSquared)
                    {
                        texture.SetPixel(x, y, color);
                    }
                }
            }

            if (apply)
            {
                texture.Apply(false, false);
            }
        }

        private void EnsureTexture()
        {
            if (biomeTexture == null)
            {
                CreateTexture(activeBiome);
            }
        }

        private Color32 ResolveColor(BiomeDefinition biome)
        {
            BiomeDefinition resolvedBiome = biome != null ? biome : activeBiome;
            if (resolvedBiome != null)
            {
                return resolvedBiome.inputColor;
            }

            if (palette != null)
            {
                BiomeDefinition fallback = palette.FindFallbackBiome();
                if (fallback != null)
                {
                    return fallback.inputColor;
                }
            }

            return new Color32(127, 173, 123, 255);
        }
    }
}
