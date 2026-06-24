using UnityEngine;

namespace UnityTerrainGANerator.Biomes
{
    public sealed class MicroBiomeMap
    {
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int[] MicroBiomeIndices { get; private set; }
        public int[] MacroClassIndices { get; private set; }

        public MicroBiomeMap(int width, int height, int[] microBiomeIndices, int[] macroClassIndices)
        {
            if (width <= 0 || height <= 0)
            {
                throw new System.ArgumentOutOfRangeException(nameof(width), "Map dimensions must be positive.");
            }

            int expectedLength = width * height;
            if (microBiomeIndices == null || microBiomeIndices.Length != expectedLength)
            {
                throw new System.ArgumentException("Micro-biome map length does not match dimensions.", nameof(microBiomeIndices));
            }

            if (macroClassIndices == null || macroClassIndices.Length != expectedLength)
            {
                throw new System.ArgumentException("Macro-class map length does not match dimensions.", nameof(macroClassIndices));
            }

            Width = width;
            Height = height;
            MicroBiomeIndices = microBiomeIndices;
            MacroClassIndices = macroClassIndices;
        }

        public int GetMicroBiomeIndex(int x, int y)
        {
            return MicroBiomeIndices[ToIndex(x, y)];
        }

        public int GetMacroClassIndex(int x, int y)
        {
            return MacroClassIndices[ToIndex(x, y)];
        }

        public int SampleMicroBiomeIndex(float normalizedX, float normalizedY)
        {
            int x = Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(normalizedX) * (Width - 1)), 0, Width - 1);
            int y = Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(normalizedY) * (Height - 1)), 0, Height - 1);
            return GetMicroBiomeIndex(x, y);
        }

        public int SampleMacroClassIndex(float normalizedX, float normalizedY)
        {
            int x = Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(normalizedX) * (Width - 1)), 0, Width - 1);
            int y = Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(normalizedY) * (Height - 1)), 0, Height - 1);
            return GetMacroClassIndex(x, y);
        }

        private int ToIndex(int x, int y)
        {
            if (x < 0 || x >= Width)
            {
                throw new System.ArgumentOutOfRangeException(nameof(x));
            }

            if (y < 0 || y >= Height)
            {
                throw new System.ArgumentOutOfRangeException(nameof(y));
            }

            return y * Width + x;
        }
    }
}
