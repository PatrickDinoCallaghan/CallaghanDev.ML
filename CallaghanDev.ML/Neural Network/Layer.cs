using CallaghanDev.ML.Enums;

namespace CallaghanDev.ML
{
    public class Layer
    {
        public int Size { get; }
        public int InputSize { get; }

        public double[] Activations { get; set; }
        public double[] Biases { get; set; }
        public double[] Derivatives { get; set; } 

        public double[] Deltas { get; set; }

        public double[,] Weights { get; set; }

        public double[][] ActivationsBatch { get; set; }
        public double[][] DerivativesBatch { get; set; }
        public double[][] DeltasBatch { get; set; }

        public Layer(int inputSize, int size, ActivationType actType)
        {
            InputSize = inputSize;
            Size = size;

            Activations = new double[size];
            Biases = new double[size];
            Derivatives = new double[size];
            Deltas = new double[size];

            Weights = new double[size, inputSize];
            ActivationType = actType; 
            
            ActivationsBatch = null;
            DerivativesBatch = null;
            DeltasBatch = null;
        }

        public ActivationType ActivationType { get; }
    }
}
