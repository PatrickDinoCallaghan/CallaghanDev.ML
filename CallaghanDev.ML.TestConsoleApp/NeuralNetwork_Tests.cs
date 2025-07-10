
using CallaghanDev.ML.Enums;

namespace CallaghanDev.ML.TestConsoleApp
{
    public class TestNN
    {
        public void NeuralNetworkXorTest()
        {
            Console.WriteLine("NeuralNetworkXorTest:");
            double[][] inputs;
            double[][] expectedOutputs;
            // Create a simple synthetic dataset
            // XOR problem dataset
            inputs = new double[][]
                {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
                };

            expectedOutputs = new double[][]
            {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };

            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Normal,
                LayerWidths = new List<int> { 2, 4, 1 },
                LayerActivations = new List<ActivationType> { ActivationType.Leakyrelu, ActivationType.Leakyrelu, ActivationType.Sigmoid },
            };

            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);


            neuralNetwork.Train(inputs, expectedOutputs, 0.5f, 1000);  // Train with 1000 epochs

            // Evaluate the network with sample inputs
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = neuralNetwork.Predict(inputs[i]);
                Console.WriteLine(prediction[0]);
                int predictedLabel = prediction[0] >= 0.1 ? 1 : 0;  // Threshold at 0.5
                int expectedLabel = (int)expectedOutputs[i][0];

                Console.WriteLine($"Expected Label:{expectedLabel}, PredictedLabel:{predictedLabel}");
            }
        }


        public void NeuralNetworkAndGPUTest()
        {
            Console.WriteLine("GPU test");
            double[][] inputs = new double[][]
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            double[][] expectedOutputs = new double[][]
            {
                new double[] { 0 },
                new double[] { 0 },
                new double[] { 0 },
                new double[] { 1 }
            };


            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.GPU,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Uniform,
                LayerWidths = new List<int> { 2, 4, 1 },
                LayerActivations = new List<ActivationType> { ActivationType.Leakyrelu, ActivationType.Leakyrelu, ActivationType.Sigmoid },
            };
            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);

            // Train the neural network
            neuralNetwork.Train(inputs, expectedOutputs, 0.01f, 1000);

            // Evaluate the network with sample inputs
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = neuralNetwork.Predict(inputs[i]);
                int predictedLabel = prediction[0] >= 0.5 ? 1 : 0;
                int expectedLabel = (int)expectedOutputs[i][0];

                Console.WriteLine($"Expected Label: {expectedLabel}, Predicted Label: {predictedLabel}");
            }
        }
        public void NeuralNetworkAndCPUTest()
        {
            Console.WriteLine("CPU test");
            double[][] inputs = new double[][]
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            double[][] expectedOutputs = new double[][]
            {
                new double[] { 0 },
                new double[] { 0 },
                new double[] { 0 },
                new double[] { 1 }
            };

            // Initialize sensory neurons

            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Uniform,
                LayerWidths = new List<int> { 2, 4, 1 },
                LayerActivations = new List<ActivationType> { ActivationType.Tanh, ActivationType.Tanh, ActivationType.Tanh },
            };
            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);

            // Train the neural network
            neuralNetwork.Train(inputs, expectedOutputs, 0.01f, 1000);

            // Evaluate the network with sample inputs
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = neuralNetwork.Predict(inputs[i]);
                int predictedLabel = prediction[0] >= 0.5 ? 1 : 0;
                int expectedLabel = (int)expectedOutputs[i][0];

                Console.WriteLine($"Expected Label: {expectedLabel}, Predicted Label: {predictedLabel}");
            }
        }
        public void NeuralNetworkOrTest()
        {
            Console.WriteLine("NeuralNetworkOrTest");
            double[][] inputs = new double[][]
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            double[][] expectedOutputs = new double[][]
            {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 1 }
            };

            // Initialize sensory neurons
            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Uniform,
                LayerWidths = new List<int> { 2, 4, 1 },
                LayerActivations = new List<ActivationType> { ActivationType.Tanh, ActivationType.Tanh, ActivationType.Tanh },
            };
            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);

            // Train the neural network
            neuralNetwork.Train(inputs, expectedOutputs, 0.01f, 1000);

            // Evaluate the network with sample inputs
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = neuralNetwork.Predict(inputs[i]);
                int predictedLabel = prediction[0] >= 0.5 ? 1 : 0;
                int expectedLabel = (int)expectedOutputs[i][0];

                Console.WriteLine($"Expected Label: {expectedLabel}, Predicted Label: {predictedLabel}");
            }
        }

    }
}