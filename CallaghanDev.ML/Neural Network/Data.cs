using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public class Data
    {
        public Layer[] layers { get; set; }
        public Parameters parameters { get; set; }

        private Random _random;

        public Data() { }
        public Data(Parameters InParameters)
        {
            this.parameters = InParameters;
            InitializeData();
        }

        /// <summary>
        /// Initializes Layers[] according to parameters and the number of sensing inputs.
        /// If loadingFromFile is true, skips randomizing weights & biases.
        /// </summary>
        public void InitializeData()
        {
            _random = new Random();

            var sizes = parameters.LayerWidths;

            layers = new Layer[sizes.Count];

            for (int i = 0; i < sizes.Count; i++)
            {
                int layerSize = sizes[i];
                int fanIn = (i == 0) ? 0 : sizes[i - 1];
                int fanOut = (i < sizes.Count - 1) ? sizes[i + 1] : 0;

                var layer = new Layer(fanIn, layerSize, parameters.LayerActivations[i]);

                if (i > 0)
                {
                    for (int neuron = 0; neuron < layerSize; neuron++)
                    {
                        layer.Biases[neuron] = 0.0f;

                        for (int prev = 0; prev < fanIn; prev++)
                        {
                            layer.Weights[neuron, prev] = Initializer(parameters.ActivationDistribution, parameters.LayerActivations[i], fanIn, fanOut);
                        }
                    }
                }

                layers[i] = layer;
            }
        }

        private float Initializer(ActivationDistribution activationDistribution, ActivationType activationType, int incomingNeurites, int outgoingNeurites)
        {
            switch (activationDistribution)
            {
                case ActivationDistribution.Normal:
                    return InitializerNormal(activationType, incomingNeurites, outgoingNeurites);
                case ActivationDistribution.Uniform:
                    return InitializerUniform(activationType, incomingNeurites, outgoingNeurites);
                default:
                    throw new NotImplementedException("Activation distribution not Implemented!");
            }

        }
        private float InitializerNormal(ActivationType activationType, int incomingNeurites, int outgoingNeurites)
        {
            switch (activationType)
            {
                case ActivationType.Tanh:
                    return XavierGlorotNormalInitializer(incomingNeurites, outgoingNeurites);
                case ActivationType.Sigmoid:
                    return XavierGlorotNormalInitializer(incomingNeurites, outgoingNeurites);
                case ActivationType.Leakyrelu:
                    return HeNormalInitializer(incomingNeurites);
                case ActivationType.Relu:
                    return HeNormalInitializer(incomingNeurites);
                default:
                    return GetRandomDouble(_random, -1, 1);
            }
        }
        private float InitializerUniform(ActivationType activationType, int incomingNeurites, int outgoingNeurites)
        {
            switch (activationType)
            {
                case ActivationType.Tanh:
                    return XavierGlorotUniformInitializer(incomingNeurites, outgoingNeurites);
                case ActivationType.Sigmoid:
                    return XavierGlorotUniformInitializer(incomingNeurites, outgoingNeurites);
                case ActivationType.Leakyrelu:
                    return HeUniformInitializer(incomingNeurites);
                case ActivationType.Relu:
                    return HeUniformInitializer(incomingNeurites);
                default:
                    return GetRandomDouble(_random, -1, 1);
            }
        }
        /// <summary>
        /// He Initialization : ReLU / LeakyReLU Linear
        /// </summary>
        /// <param name="incomingNeurites"></param>
        /// <returns></returns>
        private float HeNormalInitializer(int incomingNeurites)
        {
            float standardDeviation = MathF.Sqrt(2.0f / incomingNeurites);
            // Generate a normally distributed _random value
            float u1 = 1.0f - _random.NextSingle(); // Uniform (0,1] _random float
            float u2 = 1.0f - _random.NextSingle(); // Uniform (0,1] _random float
            float z = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(2.0f * MathF.PI * u2); // Box-Muller transform
            return z * standardDeviation; // Scale by standard deviation
        }
        /// <summary>
        /// Generates a weight using Xavier/Glorot Initialization. : Sigmoid/Tanh:
        /// </summary>
        /// <param name="incomingNeurites">Number of incoming connections (neurons in the previous layer).</param>
        /// <param name="outgoingNeurites">Number of outgoing connections (neurons in the next layer).</param>
        /// <returns>A weight value initialized using Xavier/Glorot method.</returns>
        public float XavierGlorotNormalInitializer(int incomingNeurites, int outgoingNeurites)
        {
            if (incomingNeurites <= 0)
                throw new ArgumentException("Neurites count must be positive.");

            // Compute the standard deviation for Xavier Initialization
            float standardDeviation = MathF.Sqrt(2.0f / (incomingNeurites + outgoingNeurites));

            // Box-Muller transform to generate normally distributed _random value
            float u1 = 1.0f - _random.NextSingle(); // Uniform (0,1] _random float
            float u2 = 1.0f - _random.NextSingle(); // Uniform (0,1] _random float
            float z = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2); // Normally distributed _random value

            // Scale the value by the standard deviation
            return z * standardDeviation;
        }
        /// <summary>
        /// He Initialization using uniform distribution for ReLU / LeakyReLU activations.
        /// The weights are sampled from a uniform distribution in the range [-limit, limit].
        /// </summary>
        /// <param name="incomingNeurites">Number of incoming connections (neurons in the previous layer).</param>
        /// <returns>A weight value initialized using He Uniform method.</returns>
        private float HeUniformInitializer(int incomingNeurites)
        {
            if (incomingNeurites <= 0)
                throw new ArgumentException("Neurites count must be positive.", nameof(incomingNeurites));

            // Compute the limit for the uniform distribution
            float limit = MathF.Sqrt(6.0f / incomingNeurites);

            // Generate a uniformly distributed _random value in the range [-limit, limit]
            return _random.NextSingle() * (2 * limit) - limit;
        }

        /// <summary>
        /// Xavier/Glorot Initialization using uniform distribution for Sigmoid/Tanh activations.
        /// The weights are sampled from a uniform distribution in the range [-limit, limit].
        /// </summary>
        /// <param name="incomingNeurites">Number of incoming connections (neurons in the previous layer).</param>
        /// <param name="outgoingNeurites">Number of outgoing connections (neurons in the next layer).</param>
        /// <returns>A weight value initialized using Xavier/Glorot Uniform method.</returns>
        public float XavierGlorotUniformInitializer(int incomingNeurites, int outgoingNeurites)
        {
            if (incomingNeurites <= 0)
                throw new ArgumentException("Neurites count must be positive.");

            // Compute the limit for the uniform distribution
            float limit = MathF.Sqrt(6.0f / (incomingNeurites + outgoingNeurites));

            // Generate a uniformly distributed _random value in the range [-limit, limit]
            return _random.NextSingle() * (2 * limit) - limit;
        }
        private float GetRandomDouble(Random random, float min, float max)
        {
            if (min >= max)
            {
                throw new ArgumentException("The minimum value must be less than the maximum value.");
            }

            // Generate a _random float between 0.0 (inclusive) and 1.0 (exclusive)
            float randomValue = random.NextSingle();

            // Scale and shift the value to the specified range
            return min + (randomValue * (max - min));
        }

        #region IO

        public void Save(string filePath)
        {
            using var fs = File.Open(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
            using var writer = new BinaryWriter(fs);

            // --- Layers (unchanged) ---
            writer.Write(layers.Length);
            foreach (var layer in layers)
            {
                writer.Write(layer.InputSize);
                writer.Write(layer.Size);
                writer.Write((int)layer.ActivationType);
                WriteDoubleArray(writer, layer.Activations);
                WriteDoubleArray(writer, layer.Biases);
                WriteDoubleArray(writer, layer.Derivatives);
                WriteDoubleArray(writer, layer.Deltas);
                for (int i = 0; i < layer.Size; i++)
                    for (int j = 0; j < layer.InputSize; j++)
                        writer.Write(layer.Weights[i, j]);
            }

            // --- Parameters primitives first ---
            writer.Write((int)parameters.AccelerationType);
            writer.Write((int)parameters.CostFunction);
            writer.Write((int)parameters.ActivationDistribution);
            writer.Write(parameters.L2RegulationLamda);
            writer.Write(parameters.GradientClippingThreshold);
            writer.Write(parameters.HuberLossDelta);
            writer.Write(parameters.GradientExplosionThreshold);
            writer.Write(parameters.GradientVanishingThreshold);

            // --- Then lists in that same order ---
            writer.Write(parameters.LayerWidths.Count);
            foreach (var w in parameters.LayerWidths)
                writer.Write(w);

            writer.Write(parameters.LayerActivations.Count);
            foreach (var a in parameters.LayerActivations)
                writer.Write((int)a);

            if (parameters.inputActivationMin == null || parameters.inputActivationMax == null)
                throw new InvalidOperationException("inputActivationMin/Max must be non-null before saving.");

            WriteDoubleArray(writer, parameters.inputActivationMin);
            WriteDoubleArray(writer, parameters.inputActivationMax);
        }

        public static Data Load(string filePath)
        {
            using var fs = File.Open(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(fs);

            var result = new Data();

            // --- Layers (unchanged) ---
            int layerCount = reader.ReadInt32();
            var layers = new Layer[layerCount];
            for (int idx = 0; idx < layerCount; idx++)
            {
                int inSize = reader.ReadInt32();
                int size = reader.ReadInt32();
                var actType = (ActivationType)reader.ReadInt32();
                var layer = new Layer(inSize, size, actType);

                ReadDoubleArray(reader, layer.Activations);
                ReadDoubleArray(reader, layer.Biases);
                ReadDoubleArray(reader, layer.Derivatives);
                ReadDoubleArray(reader, layer.Deltas);

                for (int i = 0; i < size; i++)
                    for (int j = 0; j < inSize; j++)
                        layer.Weights[i, j] = reader.ReadSingle();

                layers[idx] = layer;
            }
            result.layers = layers;

            var p = new Parameters
            {
                AccelerationType = (AccelerationType)reader.ReadInt32(),
                CostFunction = (CostFunctionType)reader.ReadInt32(),
                ActivationDistribution = (ActivationDistribution)reader.ReadInt32(),
                L2RegulationLamda = reader.ReadSingle(),
                GradientClippingThreshold = reader.ReadSingle(),
                HuberLossDelta = reader.ReadSingle(),
                GradientExplosionThreshold = reader.ReadSingle(),
                GradientVanishingThreshold = reader.ReadSingle()
            };

            int lwCount = reader.ReadInt32();
            p.LayerWidths = new List<int>(lwCount);
            for (int i = 0; i < lwCount; i++)
                p.LayerWidths.Add(reader.ReadInt32());

            int laCount = reader.ReadInt32();
            p.LayerActivations = new List<ActivationType>(laCount);
            for (int i = 0; i < laCount; i++)
                p.LayerActivations.Add((ActivationType)reader.ReadInt32());

            int minCount = reader.ReadInt32();
            p.inputActivationMin = new float[minCount];
            for (int i = 0; i < minCount; i++)
                p.inputActivationMin[i] = reader.ReadSingle();

            int maxCount = reader.ReadInt32();
            p.inputActivationMax = new float[maxCount];
            for (int i = 0; i < maxCount; i++)
                p.inputActivationMax[i] = reader.ReadSingle();

            result.parameters = p;
            return result;
        }

        private static void WriteDoubleArray(BinaryWriter writer, float[] arr)
        {
            writer.Write(arr.Length);
            foreach (var v in arr) writer.Write(v);
        }
        private static void ReadDoubleArray(BinaryReader reader, float[] arr)
        {
            int len = reader.ReadInt32();
            if (len != arr.Length)
            {
                throw new InvalidDataException($"Array length mismatch: expected {arr.Length}, got {len}");
            }
            for (int i = 0; i < len; i++)
            {
                arr[i] = reader.ReadSingle();
            }
        }

        #endregion
    }
}

