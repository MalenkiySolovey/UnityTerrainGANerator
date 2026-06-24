using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace UnityTerrainGANerator.Inference
{
    public enum LatentNoiseDistribution
    {
        Uniform01,
        Normal
    }

    public sealed class ONNXInferenceController : MonoBehaviour, IDisposable
    {
        public ModelAsset modelAsset;
        public BackendType backendType = BackendType.CPU;
        public string biomeInputName = "biome_onehot";
        public string noiseInputName = "noise";
        public string terrainOutputName = "terrain";
        public bool useNoiseInputs = false;
        public int noiseSeed = 12345;
        [Range(0f, 1f)] public float noiseStrength = 1f;
        public LatentNoiseDistribution noiseDistribution = LatentNoiseDistribution.Uniform01;
        public int noiseDim = 16;
        public bool useTrainingNoiseInputs = true;
        public string trainingNoiseInputPrefix = "train_noise_";
        [Range(0f, 1f)] public float trainingNoiseStrength = 1f;
        public int inputResolution = 256;
        public int macroClassCount = 7;
        public int modelInputChannels = 11;
        public int generatedHeightChannel = 3;

        private Model runtimeModel;
        private Worker worker;
        private ModelAsset loadedModelAsset;
        private BackendType loadedBackendType;

        public void Initialize()
        {
            if (worker != null && loadedModelAsset == modelAsset && loadedBackendType == backendType)
            {
                return;
            }

            ResetWorker();

            if (modelAsset == null)
            {
                throw new InvalidOperationException("ONNX ModelAsset is not assigned.");
            }

            runtimeModel = ModelLoader.Load(modelAsset);
            worker = new Worker(runtimeModel, backendType);
            loadedModelAsset = modelAsset;
            loadedBackendType = backendType;
        }

        public void Warmup()
        {
            Initialize();
            using Tensor<float> input = new Tensor<float>(
                new TensorShape(1, modelInputChannels, inputResolution, inputResolution),
                new float[modelInputChannels * inputResolution * inputResolution]
            );
            using Tensor<float> output = RunTerrain(input, null);
        }

        public Tensor<float> RunTerrain(Tensor<float> modelInputTensor, Tensor<float> noiseTensor)
        {
            Initialize();
            ValidateModelInputTensor(modelInputTensor);
            bool expectsNoise = HasModelInput(noiseInputName);
            int batchSize = modelInputTensor.shape[0];
            int resolvedNoiseSeed = useNoiseInputs ? NormalizeSeed(noiseSeed) : 0;
            using Tensor<float> ownedNoise = expectsNoise && noiseTensor == null
                ? CreateNoiseTensor(batchSize, resolvedNoiseSeed, useNoiseInputs)
                : null;
            List<Tensor<float>> ownedTrainingNoises = null;

            try
            {
                worker.SetInput(biomeInputName, modelInputTensor);
                if (expectsNoise)
                {
                    worker.SetInput(noiseInputName, noiseTensor ?? ownedNoise);
                }

                ownedTrainingNoises = SetTrainingNoiseInputs(batchSize, resolvedNoiseSeed);

                worker.Schedule();

                Tensor<float> output = PeekTerrainOutput();
                if (output == null)
                {
                    throw new InvalidOperationException("Model output tensor is not a float tensor.");
                }

                return new Tensor<float>(output.shape, output.DownloadToArray());
            }
            finally
            {
                if (ownedTrainingNoises != null)
                {
                    foreach (Tensor<float> tensor in ownedTrainingNoises)
                    {
                        tensor.Dispose();
                    }
                }
            }
        }

        public Tensor<float> RunHeight(Tensor<float> modelInputTensor, Tensor<float> noiseTensor)
        {
            Tensor<float> terrainTensor = RunTerrain(modelInputTensor, noiseTensor);
            if (terrainTensor.shape[1] == 1)
            {
                return terrainTensor;
            }

            try
            {
                return ExtractChannel(terrainTensor, Mathf.Clamp(generatedHeightChannel, 0, terrainTensor.shape[1] - 1));
            }
            finally
            {
                terrainTensor.Dispose();
            }
        }

        [ContextMenu("Use v12.4 Terrain4 ONNX Settings")]
        public void ConfigureForV124Terrain4Model()
        {
            biomeInputName = "biome_onehot";
            noiseInputName = "noise";
            terrainOutputName = "terrain";
            useNoiseInputs = true;
            noiseDistribution = LatentNoiseDistribution.Uniform01;
            noiseStrength = 1f;
            noiseDim = 16;
            useTrainingNoiseInputs = true;
            trainingNoiseInputPrefix = "train_noise_";
            trainingNoiseStrength = 1f;
            inputResolution = 256;
            macroClassCount = 7;
            modelInputChannels = 11;
            generatedHeightChannel = 3;
        }

        private void ValidateModelInputTensor(Tensor<float> modelInputTensor)
        {
            if (modelInputTensor == null)
            {
                throw new ArgumentNullException(nameof(modelInputTensor));
            }

            int expectedChannels = Mathf.Max(1, modelInputChannels);
            if (modelInputTensor.shape.rank != 4)
            {
                throw new InvalidOperationException($"Model input tensor must be BCHW rank 4, got rank {modelInputTensor.shape.rank}.");
            }

            if (modelInputTensor.shape[1] != expectedChannels)
            {
                throw new InvalidOperationException(
                    $"Model input tensor has {modelInputTensor.shape[1]} channel(s), but ONNXInferenceController expects {expectedChannels}. " +
                    "Use 11 for v12.4 terrain4 models."
                );
            }
        }

        public Tensor<float> CreateNoiseTensor(int batchSize)
        {
            return CreateNoiseTensor(batchSize, noiseSeed, useNoiseInputs);
        }

        public Tensor<float> CreateNoiseTensor(int batchSize, int seed, bool randomize)
        {
            int count = batchSize * noiseDim;
            float[] data = randomize
                ? CreateNoiseData(count, seed, noiseStrength, noiseDistribution)
                : new float[count];
            return new Tensor<float>(new TensorShape(batchSize, noiseDim), data);
        }

        private List<Tensor<float>> SetTrainingNoiseInputs(int batchSize, int baseSeed)
        {
            if (runtimeModel == null || string.IsNullOrWhiteSpace(trainingNoiseInputPrefix))
            {
                return null;
            }

            List<Tensor<float>> ownedTensors = null;
            int trainingNoiseIndex = 0;
            foreach (Model.Input input in runtimeModel.inputs)
            {
                if (!IsTrainingNoiseInput(input.name))
                {
                    continue;
                }

                int seed = OffsetSeed(baseSeed, trainingNoiseIndex + 1);
                Tensor<float> tensor = CreateTrainingNoiseTensor(input, batchSize, seed, useNoiseInputs && useTrainingNoiseInputs);
                if (ownedTensors == null)
                {
                    ownedTensors = new List<Tensor<float>>();
                }
                ownedTensors.Add(tensor);
                worker.SetInput(input.name, tensor);
                trainingNoiseIndex++;
            }

            return ownedTensors;
        }

        private Tensor<float> CreateTrainingNoiseTensor(Model.Input input, int batchSize, int seed, bool randomize)
        {
            if (!input.shape.IsStatic())
            {
                throw new InvalidOperationException(
                    $"Training noise input '{input.name}' must have a static shape in the ONNX model."
                );
            }

            TensorShape shape = input.shape.ToTensorShape();
            if (shape.rank != 4)
            {
                throw new InvalidOperationException(
                    $"Training noise input '{input.name}' must be BCHW rank 4, got rank {shape.rank}."
                );
            }

            if (shape[0] != batchSize)
            {
                throw new InvalidOperationException(
                    $"Training noise input '{input.name}' has batch {shape[0]}, but model input batch is {batchSize}."
                );
            }

            float[] data = randomize
                ? CreateNormalData(shape.length, seed, Mathf.Clamp01(trainingNoiseStrength))
                : new float[shape.length];
            return new Tensor<float>(shape, data);
        }

        public void Dispose()
        {
            ResetWorker();
        }

        public void ResetWorker()
        {
            Worker currentWorker = worker;
            worker = null;
            if (currentWorker != null)
            {
                currentWorker.Dispose();
                GC.SuppressFinalize(currentWorker);
            }

            runtimeModel = null;
            loadedModelAsset = null;
        }

        private void OnDestroy()
        {
            Dispose();
        }

        private void OnDisable()
        {
            Dispose();
        }

        private void OnApplicationQuit()
        {
            Dispose();
        }

        private Tensor<float> PeekTerrainOutput()
        {
            if (!string.IsNullOrWhiteSpace(terrainOutputName))
            {
                foreach (Model.Output output in runtimeModel.outputs)
                {
                    if (output.name == terrainOutputName)
                    {
                        return worker.PeekOutput(terrainOutputName) as Tensor<float>;
                    }
                }
            }

            return worker.PeekOutput() as Tensor<float>;
        }

        private bool HasModelInput(string inputName)
        {
            if (string.IsNullOrWhiteSpace(inputName) || runtimeModel == null)
            {
                return false;
            }

            foreach (Model.Input input in runtimeModel.inputs)
            {
                if (input.name == inputName)
                {
                    return true;
                }
            }

            return false;
        }

        private bool IsTrainingNoiseInput(string inputName)
        {
            return !string.IsNullOrWhiteSpace(inputName)
                && !string.IsNullOrWhiteSpace(trainingNoiseInputPrefix)
                && inputName.StartsWith(trainingNoiseInputPrefix, StringComparison.Ordinal);
        }

        private static Tensor<float> ExtractChannel(Tensor<float> source, int channel)
        {
            int batch = source.shape[0];
            int channels = source.shape[1];
            int height = source.shape[2];
            int width = source.shape[3];
            float[] sourceData = source.DownloadToArray();
            float[] channelData = new float[batch * height * width];

            for (int b = 0; b < batch; b++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int sourceIndex = (((b * channels + channel) * height + y) * width) + x;
                        int targetIndex = ((b * height + y) * width) + x;
                        channelData[targetIndex] = sourceData[sourceIndex];
                    }
                }
            }

            return new Tensor<float>(new TensorShape(batch, 1, height, width), channelData);
        }

        private static float[] CreateNoiseData(
            int count,
            int seed,
            float strength,
            LatentNoiseDistribution distribution)
        {
            float clampedStrength = Mathf.Clamp01(strength);
            return distribution == LatentNoiseDistribution.Normal
                ? CreateNormalData(count, seed, clampedStrength)
                : CreateUniform01Data(count, seed, clampedStrength);
        }

        private static float[] CreateUniform01Data(int count, int seed, float strength)
        {
            float[] data = new float[count];
            System.Random random = new System.Random(NormalizeSeed(seed));
            for (int index = 0; index < count; index++)
            {
                data[index] = (float)random.NextDouble() * strength;
            }

            return data;
        }

        private static float[] CreateNormalData(int count, int seed, float scale)
        {
            float[] data = new float[count];
            System.Random random = new System.Random(NormalizeSeed(seed));
            for (int index = 0; index < count; index += 2)
            {
                double u1 = System.Math.Max(random.NextDouble(), 1e-7);
                double u2 = random.NextDouble();
                double radius = System.Math.Sqrt(-2.0 * System.Math.Log(u1));
                double angle = 2.0 * System.Math.PI * u2;
                data[index] = (float)(radius * System.Math.Cos(angle)) * scale;
                if (index + 1 < count)
                {
                    data[index + 1] = (float)(radius * System.Math.Sin(angle)) * scale;
                }
            }

            return data;
        }

        private static int NormalizeSeed(int seed)
        {
            if (seed == int.MinValue)
            {
                return int.MaxValue;
            }

            return Mathf.Abs(seed);
        }

        private static int OffsetSeed(int seed, int salt)
        {
            unchecked
            {
                int mixed = (seed + 0x2C1B3C6D) ^ (salt * 0x45D9F3B);
                mixed ^= mixed >> 16;
                mixed *= 0x45D9F3B;
                mixed ^= mixed >> 16;
                return NormalizeSeed(mixed);
            }
        }

    }
}
