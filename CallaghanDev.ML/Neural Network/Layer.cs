using CallaghanDev.ML.Enums;
using Newtonsoft.Json;

namespace CallaghanDev.ML
{
    public class Layer
    {
        public int Size { get; }
        public int InputSize { get; }

        public float[] Activations { get; set; }
        public float[] Biases { get; set; }
        public float[] Derivatives { get; set; } 

        public float[] Deltas { get; set; }

        public float[,] Weights { get; set; }

        [JsonIgnore]
        public float[][] ActivationsBatch { get; set; }

        [JsonIgnore]
        public float[][] DerivativesBatch { get; set; }

        [JsonIgnore]
        public float[][] DeltasBatch { get; set; }
        public Layer() { }

        public Layer(int inputSize, int size, ActivationType actType)
        {
            InputSize = inputSize;
            Size = size;

            Activations = new float[size];
            Biases = new float[size];
            Derivatives = new float[size];
            Deltas = new float[size];

            Weights = new float[size, inputSize];
            ActivationType = actType; 
            
            ActivationsBatch = null;
            DerivativesBatch = null;
            DeltasBatch = null;
        }

        public ActivationType ActivationType { get; }
    }
}
