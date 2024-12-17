using CallaghanDev.ML.NN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.TestConsoleApp
{
    public class TestNN
    {
        public static void Run()
        {
            // Load the MNIST data
            List<double[]> inputs = new List<double[]>();
            List<double[]> expectedOutputs = new List<double[]>();

            using (var reader = new StreamReader(@"mnist_train.csv"))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');

                    // Extract label and convert to one-hot encoding
                    int label = Convert.ToInt32(values[0]);
                    double[] output = new double[10];
                    output[label] = 1;
                    expectedOutputs.Add(output);

                    // Convert pixel values to double and normalize (if required)
                    double[] input = values.Skip(1).Select(str => Convert.ToDouble(str) / 255).ToArray();  // Normalization step
                    inputs.Add(input);
                }
            }

            List<SensoryNeuron> InputNeurons = new List<SensoryNeuron>();
            for (int i = 0; i < 784; i++)
            {

                InputNeurons.Add(new SensoryNeuron(255, 0));
            }

            Parameters parameters = new Parameters()
            {
                AccelerationType= AccelerationType.GPU,
                SensoryNeurons = InputNeurons.ToArray(),
                NoHiddenLayers = 10,
                HiddenLayerWidth = 784,
                NumberOfOutputs = 10,
                DefaultActivationType = ActivationType.Leakyrelu,
                CostFunction = CostFunctionType.mse
            };

            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);  // Correct input size for MNIST

            // Assuming you want to train on a small subset for testing
            int trainSize = 1000; // Small number for demonstration; typically use more
            double[][] trainingInputs = inputs.Take(trainSize).ToArray();
            double[][] trainingOutputs = expectedOutputs.Take(trainSize).ToArray();

            // Train the neural network
            neuralNetwork.Train(trainingInputs, trainingOutputs, 0.01f, 15);

            // Evaluate network with a sample input and display output
            double[] sampleInput = trainingInputs[0]; // Using the first input for testing
            double[] prediction = neuralNetwork.Predict(sampleInput);

            // Print prediction results
            Console.WriteLine("Input Label: " + Array.IndexOf(trainingOutputs[0], 1));
            Console.WriteLine("Predicted Label: " + Array.IndexOf(prediction, prediction.Max()));


            NeuralNetwork.Save(neuralNetwork, DateTime.Now.ToString("ss.mm.dd.yyyy"));

            Console.ReadKey();
        }
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
            // Initialize sensory neurons
            List<SensoryNeuron> inputNeurons = new List<SensoryNeuron>();
            for (int i = 0; i < 2; i++)  // XOR problem has 2 inputs
            {
                inputNeurons.Add(new SensoryNeuron(0, 1));
            }
            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                SensoryNeurons = inputNeurons.ToArray(),
                NoHiddenLayers = 2,
                HiddenLayerWidth = 4,
                NumberOfOutputs = 1,
                DefaultActivationType = ActivationType.Leakyrelu,
                CostFunction = CostFunctionType.mse,
                GradientClippingThreshold = 1f,
                L2RegulationLamda = 0.01
            };

            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);


            neuralNetwork.Train(inputs, expectedOutputs, 0.1f, 1000);  // Train with 1000 epochs

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

            // Initialize sensory neurons
            List<SensoryNeuron> inputNeurons = new List<SensoryNeuron>();
            for (int i = 0; i < 2; i++)
            {
                inputNeurons.Add(new SensoryNeuron(0, 1));
            }

            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                SensoryNeurons = inputNeurons.ToArray(),
                NoHiddenLayers = 2,
                HiddenLayerWidth = 2,
                NumberOfOutputs = 1,
                DefaultActivationType = ActivationType.Tanh,
                CostFunction = CostFunctionType.mse,
                GradientClippingThreshold = 1
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
            List<SensoryNeuron> inputNeurons = new List<SensoryNeuron>();
            for (int i = 0; i < 2; i++)
            {
                inputNeurons.Add(new SensoryNeuron(0, 1));
            }

            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                SensoryNeurons = inputNeurons.ToArray(),
                NoHiddenLayers = 2,
                HiddenLayerWidth = 2,
                NumberOfOutputs = 1,
                DefaultActivationType = ActivationType.Tanh,
                CostFunction = CostFunctionType.mse,
                GradientClippingThreshold = 1
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
            List<SensoryNeuron> inputNeurons = new List<SensoryNeuron>();
            for (int i = 0; i < 2; i++)
            {
                inputNeurons.Add(new SensoryNeuron(0, 1));
            }

            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                SensoryNeurons = inputNeurons.ToArray(),
                NoHiddenLayers = 1,
                HiddenLayerWidth = 2,
                NumberOfOutputs = 1,
                DefaultActivationType = ActivationType.Tanh,
                CostFunction = CostFunctionType.mse,
                GradientClippingThreshold = 1
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

        public void NeuralNetworkRegressionTest()
        {
            Console.WriteLine("NeuralNetworkRegressionTest");
            double[][] inputs = new double[][]
            {
                new double[] { 1 },
                new double[] { 2 },
                new double[] { 3 },
                new double[] { 4 },
                new double[] { 5 }
            };

            double[][] expectedOutputs = new double[][]
            {
                new double[] { 2 },
                new double[] { 4 },
                new double[] { 6 },
                new double[] { 8 },
                new double[] { 10 }
            };

            // Initialize sensory neurons
            List<SensoryNeuron> inputNeurons = new List<SensoryNeuron>();
            inputNeurons.Add(new SensoryNeuron(1, 5));

            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                SensoryNeurons = inputNeurons.ToArray(),
                NoHiddenLayers = 3,
                HiddenLayerWidth = 3,
                NumberOfOutputs = 1,
                DefaultActivationType = ActivationType.Tanh,
                CostFunction = CostFunctionType.mse,
                GradientClippingThreshold = (float)0.8
            };
            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);

            // Train the neural network
            neuralNetwork.Train(inputs, expectedOutputs, 0.01f, 1000);

            // Evaluate the network with sample inputs
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = neuralNetwork.Predict(inputs[i]);
                double expectedValue = expectedOutputs[i][0];

                Console.WriteLine($"Expected Value: {expectedValue}, Predicted Value: {prediction[0]}");
            }
        }

    }
}