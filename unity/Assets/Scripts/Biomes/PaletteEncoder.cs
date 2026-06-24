using Unity.InferenceEngine;
using UnityEngine;

namespace UnityTerrainGANerator.Biomes
{
    public sealed class PaletteEncoder : MonoBehaviour
    {
        public const int V124ReliefControlChannelCount = 4;

        public BiomePalette palette;
        public float[] heightPriorByClass = { 0.05f, 0.22f, 0.36f, 0.48f, 0.66f, 0.78f, 0.90f };
        public float[] detailBudgetByClass = { 0.10f, 0.28f, 0.38f, 0.50f, 0.58f, 0.74f, 0.88f };
        public float[] boundaryCenterByClass = { 0f, 0.30f, 0.40f, 0.50f, 0.68f, 0.82f, 1f };
        public int boundaryDistanceSteps = 16;
        public int semanticComplexityKernelSize = 9;
        public int controlSmoothingKernelSize = 5;

        public MicroBiomeMap BuildMicroBiomeMap(Texture2D biomeTexture)
        {
            ValidateInputs(biomeTexture);
            return BuildMicroBiomeMap(biomeTexture, biomeTexture.width, biomeTexture.height);
        }

        public MicroBiomeMap BuildMicroBiomeMap(Texture2D biomeTexture, int targetWidth, int targetHeight)
        {
            return BuildMicroBiomeMap(biomeTexture, targetWidth, targetHeight, false, true, false);
        }

        public MicroBiomeMap BuildMicroBiomeMap(
            Texture2D biomeTexture,
            int targetWidth,
            int targetHeight,
            bool useFlipX,
            bool useFlipY,
            bool useSwapXY)
        {
            ValidateInputs(biomeTexture);

            int sourceWidth = biomeTexture.width;
            int sourceHeight = biomeTexture.height;
            int width = Mathf.Max(1, targetWidth);
            int height = Mathf.Max(1, targetHeight);
            Color32[] pixels = biomeTexture.GetPixels32();
            int[] microBiomeIndices = new int[width * height];
            int[] macroClassIndices = new int[width * height];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float u = width == 1 ? 0f : (float)x / (width - 1);
                    float v = height == 1 ? 0f : (float)y / (height - 1);
                    float sourceU = useSwapXY ? v : u;
                    float sourceV = useSwapXY ? u : v;

                    if (useFlipX)
                    {
                        sourceU = 1f - sourceU;
                    }

                    if (useFlipY)
                    {
                        sourceV = 1f - sourceV;
                    }

                    int sourceX = Mathf.Clamp(Mathf.RoundToInt(sourceU * (sourceWidth - 1)), 0, sourceWidth - 1);
                    int sourceY = Mathf.Clamp(Mathf.RoundToInt(sourceV * (sourceHeight - 1)), 0, sourceHeight - 1);
                    Color32 color = pixels[sourceY * sourceWidth + sourceX];
                    BiomeDefinition biome = palette.FindNearestBiome(color);
                    int index = y * width + x;
                    microBiomeIndices[index] = biome != null ? biome.microBiomeId : ResolveFallbackMicroBiomeId();
                    macroClassIndices[index] = palette.ResolveMacroClassIndex(biome);
                }
            }

            return new MicroBiomeMap(width, height, microBiomeIndices, macroClassIndices);
        }

        public Texture2D BuildMacroColorTexture(Texture2D biomeTexture)
        {
            return BuildMacroColorTexture(biomeTexture, biomeTexture.width, biomeTexture.height);
        }

        public Texture2D BuildMacroColorTexture(Texture2D biomeTexture, int targetWidth, int targetHeight)
        {
            MicroBiomeMap map = BuildMicroBiomeMap(biomeTexture, targetWidth, targetHeight);
            Texture2D texture = new Texture2D(map.Width, map.Height, TextureFormat.RGBA32, false);
            texture.SetPixels32(BuildMacroColorPixels(map));
            texture.Apply(false, false);
            return texture;
        }

        public Color32[] BuildMacroColorPixels(MicroBiomeMap map)
        {
            Color32[] pixels = new Color32[map.Width * map.Height];
            for (int y = 0; y < map.Height; y++)
            {
                for (int x = 0; x < map.Width; x++)
                {
                    int index = y * map.Width + x;
                    pixels[index] = palette.FindMacroColor(map.GetMacroClassIndex(x, y));
                }
            }

            return pixels;
        }

        public Tensor<float> BuildMacroOneHotTensor(Texture2D biomeTexture)
        {
            MicroBiomeMap map = BuildMicroBiomeMap(biomeTexture);
            return BuildMacroOneHotTensor(map);
        }

        public Tensor<float> BuildMacroOneHotTensor(Texture2D biomeTexture, int targetWidth, int targetHeight)
        {
            MicroBiomeMap map = BuildMicroBiomeMap(biomeTexture, targetWidth, targetHeight);
            return BuildMacroOneHotTensor(map);
        }

        public Tensor<float> BuildMacroOneHotTensor(MicroBiomeMap map)
        {
            float[] data = BuildMacroOneHotArray(map);
            TensorShape shape = new TensorShape(1, palette.macroClassCount, map.Height, map.Width);
            return new Tensor<float>(shape, data);
        }

        public Tensor<float> BuildModelInputTensor(MicroBiomeMap map, int targetChannelCount)
        {
            float[] data = BuildModelInputArray(map, targetChannelCount);
            TensorShape shape = new TensorShape(1, targetChannelCount, map.Height, map.Width);
            return new Tensor<float>(shape, data);
        }

        public float[] BuildMacroOneHotArray(Texture2D biomeTexture)
        {
            MicroBiomeMap map = BuildMicroBiomeMap(biomeTexture);
            return BuildMacroOneHotArray(map);
        }

        public float[] BuildMacroOneHotArray(MicroBiomeMap map)
        {
            int channelCount = palette.macroClassCount;
            float[] data = new float[channelCount * map.Height * map.Width];

            for (int y = 0; y < map.Height; y++)
            {
                for (int x = 0; x < map.Width; x++)
                {
                    int macroClass = map.GetMacroClassIndex(x, y);
                    int index = ((macroClass * map.Height + y) * map.Width) + x;
                    data[index] = 1f;
                }
            }

            return data;
        }

        public float[] BuildModelInputArray(MicroBiomeMap map, int targetChannelCount)
        {
            int macroChannelCount = palette.macroClassCount;
            if (targetChannelCount == macroChannelCount)
            {
                return BuildMacroOneHotArray(map);
            }

            int v124ChannelCount = macroChannelCount + V124ReliefControlChannelCount;
            if (targetChannelCount != v124ChannelCount)
            {
                throw new System.InvalidOperationException(
                    $"Unsupported model input channel count {targetChannelCount}. Expected {macroChannelCount} for legacy models or {v124ChannelCount} for v12.4 relief controls."
                );
            }

            float[] macroOneHot = BuildMacroOneHotArray(map);
            float[] data = new float[targetChannelCount * map.Height * map.Width];
            System.Array.Copy(macroOneHot, data, macroOneHot.Length);
            AppendV124ReliefControlMaps(data, macroChannelCount, map);
            return data;
        }

        private void AppendV124ReliefControlMaps(float[] data, int macroChannelCount, MicroBiomeMap map)
        {
            int width = map.Width;
            int height = map.Height;
            float[] heightPrior = SmoothControl(BuildWeightedPrior(map, heightPriorByClass), width, height, controlSmoothingKernelSize);
            float[] boundary = BuildBoundaryDistance(map, width, height);
            float[] baseDetailBudget = SmoothControl(BuildWeightedPrior(map, detailBudgetByClass), width, height, controlSmoothingKernelSize);
            float[] detailBudget = BuildDetailBudget(baseDetailBudget, boundary);
            float[] complexity = BuildSemanticComplexity(map, width, height);

            CopyChannel(heightPrior, data, macroChannelCount, width, height);
            CopyChannel(detailBudget, data, macroChannelCount + 1, width, height);
            CopyChannel(boundary, data, macroChannelCount + 2, width, height);
            CopyChannel(complexity, data, macroChannelCount + 3, width, height);
        }

        private static float[] ExtractChannel(float[] data, int channelCount, int width, int height, int channel)
        {
            float[] values = new float[width * height];
            if (channel < 0 || channel >= channelCount)
            {
                return values;
            }

            int sourceOffset = channel * width * height;
            System.Array.Copy(data, sourceOffset, values, 0, values.Length);
            return values;
        }

        private static void CopyChannel(float[] source, float[] target, int channel, int width, int height)
        {
            int offset = channel * width * height;
            for (int index = 0; index < width * height; index++)
            {
                target[offset + index] = Mathf.Clamp01(source[index]);
            }
        }

        private static float[] BuildDetailBudget(float[] baseBudget, float[] boundary)
        {
            float[] budget = new float[baseBudget.Length];
            for (int index = 0; index < budget.Length; index++)
            {
                budget[index] = Mathf.Clamp01(baseBudget[index] * (0.65f + 0.35f * boundary[index]));
            }

            return budget;
        }

        private float[] BuildWeightedPrior(MicroBiomeMap map, float[] valuesByClass)
        {
            float[] prior = new float[map.Width * map.Height];
            for (int y = 0; y < map.Height; y++)
            {
                for (int x = 0; x < map.Width; x++)
                {
                    int macroClass = map.GetMacroClassIndex(x, y);
                    prior[y * map.Width + x] = valuesByClass != null && macroClass >= 0 && macroClass < valuesByClass.Length
                        ? Mathf.Clamp01(valuesByClass[macroClass])
                        : 0f;
                }
            }

            return prior;
        }

        private float[] BuildBoundaryDistance(MicroBiomeMap map, int width, int height)
        {
            int steps = Mathf.Max(1, boundaryDistanceSteps);
            bool[] current = new bool[width * height];
            bool[] next = new bool[width * height];
            float[] distance = new float[width * height];
            for (int index = 0; index < current.Length; index++)
            {
                current[index] = true;
            }

            for (int step = 0; step < steps; step++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int index = y * width + x;
                        int macroClass = map.GetMacroClassIndex(x, y);
                        bool survives = current[index];
                        for (int oy = -1; oy <= 1 && survives; oy++)
                        {
                            int sampleY = Mathf.Clamp(y + oy, 0, height - 1);
                            for (int ox = -1; ox <= 1; ox++)
                            {
                                int sampleX = Mathf.Clamp(x + ox, 0, width - 1);
                                int sampleIndex = sampleY * width + sampleX;
                                if (!current[sampleIndex] || map.GetMacroClassIndex(sampleX, sampleY) != macroClass)
                                {
                                    survives = false;
                                    break;
                                }
                            }
                        }

                        next[index] = survives;
                    }
                }

                for (int index = 0; index < current.Length; index++)
                {
                    current[index] = next[index];
                    if (current[index])
                    {
                        distance[index] += 1f;
                    }
                }
            }

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int index = y * width + x;
                    int macroClass = map.GetMacroClassIndex(x, y);
                    float classWeight = boundaryCenterByClass != null && macroClass >= 0 && macroClass < boundaryCenterByClass.Length
                        ? Mathf.Clamp01(boundaryCenterByClass[macroClass])
                        : 0f;
                    distance[index] = Mathf.Clamp01((distance[index] / steps) * classWeight);
                }
            }

            return SmoothControl(distance, width, height, controlSmoothingKernelSize);
        }

        private float[] BuildSemanticComplexity(MicroBiomeMap map, int width, int height)
        {
            int kernelSize = Mathf.Max(1, semanticComplexityKernelSize);
            if (kernelSize % 2 == 0)
            {
                kernelSize++;
            }

            int radius = kernelSize / 2;
            int classCount = Mathf.Max(1, palette.macroClassCount);
            float logClassCount = Mathf.Log(Mathf.Max(2, classCount));
            int[] counts = new int[classCount];
            float[] complexity = new float[width * height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    System.Array.Clear(counts, 0, counts.Length);
                    int total = 0;
                    for (int oy = -radius; oy <= radius; oy++)
                    {
                        int sampleY = Mathf.Clamp(y + oy, 0, height - 1);
                        for (int ox = -radius; ox <= radius; ox++)
                        {
                            int sampleX = Mathf.Clamp(x + ox, 0, width - 1);
                            int macroClass = Mathf.Clamp(map.GetMacroClassIndex(sampleX, sampleY), 0, classCount - 1);
                            counts[macroClass]++;
                            total++;
                        }
                    }

                    float entropy = 0f;
                    for (int classIndex = 0; classIndex < classCount; classIndex++)
                    {
                        if (counts[classIndex] == 0)
                        {
                            continue;
                        }

                        float probability = counts[classIndex] / (float)total;
                        entropy -= probability * Mathf.Log(probability);
                    }

                    complexity[y * width + x] = Mathf.Clamp01(entropy / logClassCount);
                }
            }

            return SmoothControl(complexity, width, height, controlSmoothingKernelSize);
        }

        private static float[] SmoothControl(float[] source, int width, int height, int kernelSize)
        {
            int radius = Mathf.Max(0, kernelSize / 2);
            float[] smoothed = new float[width * height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float sum = 0f;
                    int count = 0;
                    for (int dy = -radius; dy <= radius; dy++)
                    {
                        int sampleY = Mathf.Clamp(y + dy, 0, height - 1);
                        for (int dx = -radius; dx <= radius; dx++)
                        {
                            int sampleX = Mathf.Clamp(x + dx, 0, width - 1);
                            sum += source[sampleY * width + sampleX];
                            count++;
                        }
                    }

                    smoothed[y * width + x] = count > 0 ? sum / count : source[y * width + x];
                }
            }

            return smoothed;
        }

        private void ValidateInputs(Texture2D biomeTexture)
        {
            if (palette == null)
            {
                throw new System.InvalidOperationException("Biome palette is not assigned.");
            }

            if (biomeTexture == null)
            {
                throw new System.ArgumentNullException(nameof(biomeTexture));
            }

            if (!biomeTexture.isReadable)
            {
                throw new System.InvalidOperationException("Biome texture must be readable for GetPixels32(). In Unity enable Read/Write for this texture, or use Tools -> Terrain GAN -> Preview Window, which can fix imported textures automatically.");
            }
        }

        private int ResolveFallbackMicroBiomeId()
        {
            BiomeDefinition fallback = palette.FindFallbackBiome();
            return fallback == null ? -1 : fallback.microBiomeId;
        }
    }
}
