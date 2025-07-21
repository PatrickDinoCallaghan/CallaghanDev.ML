using CallaghanDev.ML.Enums;
using Newtonsoft.Json;
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

        [JsonIgnore]
        private Random _random;

        public Data()
        {
            _random = new Random();
        }
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
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.Auto,
                Formatting = Formatting.Indented
            };
            string json = JsonConvert.SerializeObject(this, settings);
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Load Data from JSON (written by SaveJson).
        /// </summary>
        public static Data Load(string filePath)
        {
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.Auto
            };
            string json = File.ReadAllText(filePath);
            var data = JsonConvert.DeserializeObject<Data>(json, settings);

            // re‑init anything Json skipped
            data._random = new Random();

            return data;
        }
        #endregion
    }
}

