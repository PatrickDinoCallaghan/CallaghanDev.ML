﻿
using CallaghanDev.ML.Enums;

namespace CallaghanDev.ML.TestConsoleApp
{
    public class TestNN
    {
        public void NeuralNetworkXorTest()
        {
            Console.WriteLine("NeuralNetworkXorTest:");
            float[][] inputs;
            float[][] expectedOutputs;
            // Create a simple synthetic dataset
            // XOR problem dataset
            inputs = new float[][]
                {
                new float[] { 0, 0 },
                new float[] { 0, 1 },
                new float[] { 1, 0 },
                new float[] { 1, 1 }
                };

            expectedOutputs = new float[][]
            {
                new float[] { 0 },
                new float[] { 1 },
                new float[] { 1 },
                new float[] { 0 }
            };

            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Normal,
                LayerWidths = new List<int> { 2, 4, 8, 4, 1 },
                LayerActivations = new List<ActivationType> { ActivationType.Leakyrelu, ActivationType.Leakyrelu, ActivationType.Leakyrelu, ActivationType.Leakyrelu, ActivationType.Sigmoid },
            };

            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);


            neuralNetwork.Train(inputs, expectedOutputs, 0.5f, 1000);  // Train with 1000 epochs

            Console.WriteLine($"\n");
            // Evaluate the network with sample inputs
            for (int i = 0; i < inputs.Length; i++)
            {
                float[] prediction = neuralNetwork.Predict(inputs[i]);
                Console.WriteLine(prediction[0]);
                int predictedLabel = prediction[0] >= 0.1 ? 1 : 0;  // Threshold at 0.5
                int expectedLabel = (int)expectedOutputs[i][0];

                Console.WriteLine($"Expected Label:{expectedLabel}, PredictedLabel:{predictedLabel}");
            }
        }
        public void NeuralNetworkAndGPUTest()
        {
            Console.WriteLine("GPU test");

            float[][] inputs = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 0, 1 },
                new float[] { 1, 0 },
                new float[] { 1, 1 }
            };

            float[][] expectedOutputs = new float[][]
            {
                new float[] { 0 },
                new float[] { 0 },
                new float[] { 0 },
                new float[] { 1 }
            };


            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.GPU,
                CostFunction = CostFunctionType.huberLoss,
                HuberLossDelta = 0.5f,
                ActivationDistribution = ActivationDistribution.Uniform,
                LayerWidths = new List<int> { 2, 4, 8, 4, 1 },
                LayerActivations = new List<ActivationType> { ActivationType.Leakyrelu, ActivationType.Leakyrelu, ActivationType.Leakyrelu, ActivationType.Leakyrelu, ActivationType.None },

            };

            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);


            neuralNetwork.Train(inputs, expectedOutputs, 0.01f, 1000);

            Console.WriteLine($"\n");

            for (int i = 0; i < inputs.Length; i++)
            {
                float[] prediction = neuralNetwork.Predict(inputs[i]);
                int predictedLabel = prediction[0] >= 0.5 ? 1 : 0;
                int expectedLabel = (int)expectedOutputs[i][0];

                Console.WriteLine($"Expected Label: {expectedLabel}, Predicted Label: {predictedLabel}");
            }
        }
        public void NeuralNetworkAndCPUTest()
        {
            Console.WriteLine("CPU test");
            float[][] inputs = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 0, 1 },
                new float[] { 1, 0 },
                new float[] { 1, 1 }
            };

            float[][] expectedOutputs = new float[][]
            {
                new float[] { 0 },
                new float[] { 0 },
                new float[] { 0 },
                new float[] { 1 }
            };

            // Initialize sensory neurons

            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Uniform,
                LayerWidths = new List<int> { 2, 4, 8, 4, 1 },
                LayerActivations = new List<ActivationType> { ActivationType.Tanh, ActivationType.Tanh, ActivationType.Tanh, ActivationType.Tanh, ActivationType.Tanh },

            };
            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);

            // Train the neural network
            neuralNetwork.Train(inputs, expectedOutputs, 0.01f, 1000);

            Console.WriteLine($"\n");
            // Evaluate the network with sample inputs
            for (int i = 0; i < inputs.Length; i++)
            {
                float[] prediction = neuralNetwork.Predict(inputs[i]);
                int predictedLabel = prediction[0] >= 0.5 ? 1 : 0;
                int expectedLabel = (int)expectedOutputs[i][0];

                Console.WriteLine($"Expected Label: {expectedLabel}, Predicted Label: {predictedLabel}");
            }
        }
        public void NeuralNetworkOrTest()
        {
            Console.WriteLine("NeuralNetworkOrTest");
            float[][] inputs = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 0, 1 },
                new float[] { 1, 0 },
                new float[] { 1, 1 }
            };

            float[][] expectedOutputs = new float[][]
            {
                new float[] { 0 },
                new float[] { 1 },
                new float[] { 1 },
                new float[] { 1 }
            };


            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Uniform,
                LayerWidths = new List<int> { 2, 4,4, 1 },
                LayerActivations = new List<ActivationType> { ActivationType.Tanh, ActivationType.Tanh, ActivationType.Tanh, ActivationType.Tanh },
            };

            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);


            neuralNetwork.Train(inputs, expectedOutputs, 0.01f, 1000);

            Console.WriteLine($"\n");

            for (int i = 0; i < inputs.Length; i++)
            {
                float[] prediction = neuralNetwork.Predict(inputs[i]);
                int predictedLabel = prediction[0] >= 0.5 ? 1 : 0;
                int expectedLabel = (int)expectedOutputs[i][0];

                Console.WriteLine($"Expected Label: {expectedLabel}, Predicted Label: {predictedLabel}");
            }
        }
        public void NeuralNetworkBatchXorTestCUDA()
        {
            Console.WriteLine("NeuralNetworkBatchXorTestCUDA:");

            float[][] inputs = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 0, 1 },
                new float[] { 1, 0 },
                new float[] { 1, 1 }
            };
            float[][] expectedOutputs = new float[][]
            {
                new float[] { 0 },
                new float[] { 1 },
                new float[] { 1 },
                new float[] { 0 }
            };


            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CUDA,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Normal,
                LayerWidths = new List<int> { 2, 4, 8, 4, 1 },
                LayerActivations = new List<ActivationType>
                {
                    ActivationType.Leakyrelu,
                    ActivationType.Leakyrelu,
                    ActivationType.Leakyrelu,
                    ActivationType.Leakyrelu,
                    ActivationType.Sigmoid
                },
            };

            var nn = new NeuralNetwork(parameters);


            int batchSize = 100;
            float lr = 0.1f;
            int epochs = 1000;
            nn.TrainBatch(inputs, expectedOutputs, batchSize, lr, epochs);

            Console.WriteLine();

            for (int i = 0; i < inputs.Length; i++)
            {
                var pred = nn.Predict(inputs[i])[0];
                int p = pred >= 0.5 ? 1 : 0;
                int e = (int)expectedOutputs[i][0];
                Console.WriteLine($"Input: [{inputs[i][0]}, {inputs[i][1]}]  →  Pred={pred:F3}  Label={p}  (Expected {e})");
            }
        }
    }
}