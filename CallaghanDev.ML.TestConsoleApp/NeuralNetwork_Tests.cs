using CallaghanDev.ML.NN;

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
                NoHiddenLayers = 1,
                HiddenLayerWidth = 4,
                NumberOfOutputs = 1,
                DefaultActivationType = ActivationType.Leakyrelu,
                CostFunction = CostFunctionType.mse,
                GradientClippingThreshold = 1f,
                L2RegulationLamda = 0.01
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

        public void NeuralNetworkTwoSpiralsTest()
        {
            Console.WriteLine("NeuralNetworkTwoSpiralsTest:");

            // Generate two interleaving spirals dataset
            int pointsPerSpiral = 500;
            double[][] inputs = new double[pointsPerSpiral * 2][];
            double[][] expectedOutputs = new double[pointsPerSpiral * 2][];

            // Parameters to generate the spiral
            // t ranges from 0 to a certain value (e.g., 5 rotations * 2π = ~31.4)
            double maxTheta = 5 * Math.PI;
            double step = maxTheta / (pointsPerSpiral - 1);

            int index = 0;
            for (int i = 0; i < pointsPerSpiral; i++)
            {
                double t = i * step;
                // First spiral (Class 0)
                double x1 = t * Math.Cos(t);
                double y1 = t * Math.Sin(t);
                inputs[index] = new double[] { x1, y1 };
                expectedOutputs[index] = new double[] { 0 };
                index++;

                // Second spiral (Class 1), offset by π in angle
                double x2 = t * Math.Cos(t + Math.PI);
                double y2 = t * Math.Sin(t + Math.PI);
                inputs[index] = new double[] { x2, y2 };
                expectedOutputs[index] = new double[] { 1 };
                index++;
            }

            // Normalize or scale inputs if needed (optional)
            // For example, you could divide by maxTheta to scale down values.

            // Initialize sensory neurons based on input range
            // Assuming spiral coordinates may range roughly in [-maxTheta, maxTheta]
            double inputMin = -maxTheta;
            double inputMax = maxTheta;
            List<SensoryNeuron> inputNeurons = new List<SensoryNeuron>();
            for (int i = 0; i < 2; i++)  // Spiral problem has 2 inputs (x, y)
            {
                inputNeurons.Add(new SensoryNeuron(inputMin, inputMax));
            }

            Parameters parameters = new Parameters()
            {
                AccelerationType = AccelerationType.CPU,
                SensoryNeurons = inputNeurons.ToArray(),
                NoHiddenLayers = 2,
                HiddenLayerWidth = 16,
                NumberOfOutputs = 1,
                DefaultActivationType = ActivationType.Leakyrelu,
                CostFunction = CostFunctionType.mse,
                GradientClippingThreshold = 1f,
                L2RegulationLamda = 0.01
            };

            // Setup the neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(parameters);

            // Train the network
            // Adjust learning rate, epochs as needed. Two-spirals is a tough problem.
            neuralNetwork.Train(inputs, expectedOutputs, 0.01f, 5000);

            // Evaluate on some sample points from the dataset
            // We'll just take a few random points to check how the network responds
            Random rnd = new Random(42);
            for (int i = 0; i < 10; i++)
            {
                int sampleIndex = rnd.Next(inputs.Length);
                double[] sampleInput = inputs[sampleIndex];
                double[] prediction = neuralNetwork.Predict(sampleInput);

                // Classify based on a threshold
                int predictedLabel = prediction[0] >= 0.5 ? 1 : 0;
                int expectedLabel = (int)expectedOutputs[sampleIndex][0];

                Console.WriteLine($"Input: ({sampleInput[0]:F2}, {sampleInput[1]:F2}) " +
                                  $"Expected: {expectedLabel}, Predicted: {predictedLabel}, Raw Output: {prediction[0]:F4}");
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
                NoHiddenLayers = 1,
                HiddenLayerWidth = 4,
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
                NoHiddenLayers = 1,
                HiddenLayerWidth = 4,
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
                HiddenLayerWidth = 4,
                NumberOfOutputs = 1,
                DefaultActivationType = ActivationType.Tanh,
                CostFunction = CostFunctionType.mse,
                GradientClippingThreshold = 1,
                ActivationDistribution = ActivationDistribution.Uniform,
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