using UnityEngine;

namespace UnityTerrainGANerator.TerrainGeneration
{
    public sealed class HeightPostprocessor : MonoBehaviour
    {
        [Header("Tensor")]
        public bool clamp01 = true;
        public bool resize256To257 = true;
        public int targetHeightmapResolution = 257;

        [Header("Height Range")]
        public bool remapWithPercentiles = true;
        [Range(0f, 0.25f)] public float lowerPercentile = 0.02f;
        [Range(0.75f, 1f)] public float upperPercentile = 0.98f;
        [Range(0f, 1f)] public float outputMin = 0.02f;
        [Range(0f, 1f)] public float outputMax = 0.42f;

        [Header("Smoothing")]
        [Min(0)] public int smoothingIterations = 2;
        [Range(0f, 1f)] public float smoothingStrength = 0.65f;

        private bool flipX = false;
        private bool flipY = false;

        public void ConfigureOutputMirror(bool mirrorWidth, bool mirrorLength)
        {
            flipX = mirrorLength;
            flipY = mirrorWidth;
        }

        public float[,] TensorToHeights(float[] tensorData, int width, int height)
        {
            return TensorToHeights(tensorData, width, height, 0, 1);
        }

        public float[,] TensorToHeights(float[] tensorData, int width, int height, int channelIndex, int channelCount)
        {
            if (tensorData == null)
            {
                throw new System.ArgumentNullException(nameof(tensorData));
            }

            if (channelIndex < 0 || channelIndex >= channelCount)
            {
                throw new System.ArgumentOutOfRangeException(nameof(channelIndex));
            }

            if (tensorData.Length < width * height * channelCount)
            {
                throw new System.ArgumentException("Tensor data is shorter than width * height * channelCount.", nameof(tensorData));
            }

            const bool swapXY = true;
            int outputWidth = swapXY ? height : width;
            int outputHeight = swapXY ? width : height;
            float[,] heights = new float[outputHeight, outputWidth];
            for (int y = 0; y < outputHeight; y++)
            {
                for (int x = 0; x < outputWidth; x++)
                {
                    float u = outputWidth == 1 ? 0f : (float)x / (outputWidth - 1);
                    float v = outputHeight == 1 ? 0f : (float)y / (outputHeight - 1);
                    float sourceU = swapXY ? v : u;
                    float sourceV = swapXY ? u : v;

                    if (flipX)
                    {
                        sourceU = 1f - sourceU;
                    }

                    if (flipY)
                    {
                        sourceV = 1f - sourceV;
                    }

                    int sourceX = Mathf.Clamp(Mathf.RoundToInt(sourceU * (width - 1)), 0, width - 1);
                    int sourceY = Mathf.Clamp(Mathf.RoundToInt(sourceV * (height - 1)), 0, height - 1);
                    int tensorIndex = ((channelIndex * height + sourceY) * width) + sourceX;
                    float value = tensorData[tensorIndex];
                    heights[y, x] = clamp01 ? Mathf.Clamp01(value) : value;
                }
            }

            if (remapWithPercentiles)
            {
                heights = RemapByPercentiles(heights);
            }

            if (smoothingIterations > 0)
            {
                heights = Smooth(heights, smoothingIterations, smoothingStrength);
            }

            if (resize256To257)
            {
                int targetResolution = Mathf.Max(2, targetHeightmapResolution);
                if (targetResolution != heights.GetLength(0) || targetResolution != heights.GetLength(1))
                {
                    return ResizeHeights(heights, targetResolution);
                }
            }

            return heights;
        }

        public float[,] ResizeHeights(float[,] heights, int targetResolution)
        {
            int sourceHeight = heights.GetLength(0);
            int sourceWidth = heights.GetLength(1);
            float[,] resized = new float[targetResolution, targetResolution];

            for (int y = 0; y < targetResolution; y++)
            {
                float sourceY = (float)y * (sourceHeight - 1) / (targetResolution - 1);
                int y0 = Mathf.FloorToInt(sourceY);
                int y1 = Mathf.Min(y0 + 1, sourceHeight - 1);
                float ty = sourceY - y0;

                for (int x = 0; x < targetResolution; x++)
                {
                    float sourceX = (float)x * (sourceWidth - 1) / (targetResolution - 1);
                    int x0 = Mathf.FloorToInt(sourceX);
                    int x1 = Mathf.Min(x0 + 1, sourceWidth - 1);
                    float tx = sourceX - x0;

                    float a = Mathf.Lerp(heights[y0, x0], heights[y0, x1], tx);
                    float b = Mathf.Lerp(heights[y1, x0], heights[y1, x1], tx);
                    resized[y, x] = Mathf.Lerp(a, b, ty);
                }
            }

            return resized;
        }

        private float[,] RemapByPercentiles(float[,] heights)
        {
            int height = heights.GetLength(0);
            int width = heights.GetLength(1);
            float[] values = new float[height * width];
            int index = 0;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    values[index] = heights[y, x];
                    index++;
                }
            }

            System.Array.Sort(values);
            int maxIndex = values.Length - 1;
            int lowIndex = Mathf.Clamp(Mathf.RoundToInt(maxIndex * lowerPercentile), 0, maxIndex);
            int highIndex = Mathf.Clamp(Mathf.RoundToInt(maxIndex * upperPercentile), lowIndex, maxIndex);
            float inputMin = values[lowIndex];
            float inputMax = values[highIndex];
            float denominator = Mathf.Max(inputMax - inputMin, 0.0001f);
            float minHeight = Mathf.Min(outputMin, outputMax);
            float maxHeight = Mathf.Max(outputMin, outputMax);
            float[,] remapped = new float[height, width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float normalized = Mathf.Clamp01((heights[y, x] - inputMin) / denominator);
                    remapped[y, x] = Mathf.Lerp(minHeight, maxHeight, normalized);
                }
            }

            return remapped;
        }

        private static float[,] Smooth(float[,] heights, int iterations, float strength)
        {
            int height = heights.GetLength(0);
            int width = heights.GetLength(1);
            float[,] current = heights;
            float blend = Mathf.Clamp01(strength);

            for (int iteration = 0; iteration < iterations; iteration++)
            {
                float[,] next = (float[,])current.Clone();
                for (int y = 1; y < height - 1; y++)
                {
                    for (int x = 1; x < width - 1; x++)
                    {
                        float sum = 0f;
                        for (int oy = -1; oy <= 1; oy++)
                        {
                            for (int ox = -1; ox <= 1; ox++)
                            {
                                sum += current[y + oy, x + ox];
                            }
                        }

                        float average = sum / 9f;
                        next[y, x] = Mathf.Lerp(current[y, x], average, blend);
                    }
                }

                current = next;
            }

            return current;
        }
    }
}
