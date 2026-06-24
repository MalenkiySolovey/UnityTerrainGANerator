using UnityEngine;

namespace UnityTerrainGANerator.TerrainGeneration
{
    public sealed class TerrainTexturePostprocessor : MonoBehaviour
    {
        public FilterMode filterMode = FilterMode.Bilinear;
        [Range(0f, 2f)] public float saturation = 1f;
        [Range(0f, 2f)] public float brightness = 1f;
        [Range(0f, 2f)] public float contrast = 1f;
        [Range(0.25f, 3f)] public float gamma = 1f;
        public Color tint = Color.white;

        private bool flipX = false;
        private bool flipY = false;

        public void ConfigureOutputMirror(bool mirrorWidth, bool mirrorLength)
        {
            flipX = mirrorLength;
            flipY = mirrorWidth;
        }

        public Texture2D TensorToTexture(
            float[] tensorData,
            int width,
            int height,
            int channelCount,
            int redChannel = 0,
            int greenChannel = 1,
            int blueChannel = 2)
        {
            if (tensorData == null)
            {
                throw new System.ArgumentNullException(nameof(tensorData));
            }

            ValidateChannel(redChannel, channelCount, nameof(redChannel));
            ValidateChannel(greenChannel, channelCount, nameof(greenChannel));
            ValidateChannel(blueChannel, channelCount, nameof(blueChannel));

            if (tensorData.Length < width * height * channelCount)
            {
                throw new System.ArgumentException("Tensor data is shorter than width * height * channelCount.", nameof(tensorData));
            }

            const bool swapXY = true;
            int outputWidth = swapXY ? height : width;
            int outputHeight = swapXY ? width : height;
            Texture2D texture = new Texture2D(outputWidth, outputHeight, TextureFormat.RGBA32, false, false);
            texture.name = "Generated ONNX Terrain Texture";
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.filterMode = filterMode;

            Color32[] pixels = new Color32[outputWidth * outputHeight];
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
                    float r = ReadChannel(tensorData, redChannel, sourceX, sourceY, width, height);
                    float g = ReadChannel(tensorData, greenChannel, sourceX, sourceY, width, height);
                    float b = ReadChannel(tensorData, blueChannel, sourceX, sourceY, width, height);
                    pixels[y * outputWidth + x] = AdjustColor(new Color(Mathf.Clamp01(r), Mathf.Clamp01(g), Mathf.Clamp01(b), 1f));
                }
            }

            texture.SetPixels32(pixels);
            texture.Apply(false, false);
            return texture;
        }

        private static void ValidateChannel(int channel, int channelCount, string parameterName)
        {
            if (channel < 0 || channel >= channelCount)
            {
                throw new System.ArgumentOutOfRangeException(parameterName, $"Channel {channel} is outside tensor channel count {channelCount}.");
            }
        }

        private static float ReadChannel(float[] tensorData, int channel, int x, int y, int width, int height)
        {
            return tensorData[((channel * height + y) * width) + x];
        }

        private Color AdjustColor(Color source)
        {
            float gray = source.grayscale;
            Color color = Color.Lerp(new Color(gray, gray, gray, 1f), source, Mathf.Max(0f, saturation));
            color.r = ((color.r - 0.5f) * contrast + 0.5f) * brightness * tint.r;
            color.g = ((color.g - 0.5f) * contrast + 0.5f) * brightness * tint.g;
            color.b = ((color.b - 0.5f) * contrast + 0.5f) * brightness * tint.b;

            float safeGamma = Mathf.Max(0.0001f, gamma);
            float inverseGamma = 1f / safeGamma;
            color.r = Mathf.Pow(Mathf.Clamp01(color.r), inverseGamma);
            color.g = Mathf.Pow(Mathf.Clamp01(color.g), inverseGamma);
            color.b = Mathf.Pow(Mathf.Clamp01(color.b), inverseGamma);
            color.a = 1f;
            return color;
        }
    }
}
