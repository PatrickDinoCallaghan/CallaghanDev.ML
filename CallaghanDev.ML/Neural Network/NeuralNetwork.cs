﻿using CallaghanDev.Common.Math;
using CallaghanDev.ML.Neural_Network;
using CallaghanDev.ML.Neural_Network.Exceptions;
using CallaghanDev.ML.NN;
using CallaghanDev.ML.NN.Training;
using CallaghanDev.Utilities.Math;
using CallaghanDev.Utilities.MathTools;
using Newtonsoft.Json;

namespace CallaghanDev.ML
{
    /// <summary>
    /// Represents a neural network designed for parallel processing and GPU acceleration using ILGPU.
    /// 
    /// Techniques and Features:
    /// - **Data Structure**: 
    ///   - Uses a `Matrix` to store neurons and their connections.
    ///   - `NeuriteTensor` and `NeuriteTensorT` matrices to represent connections between neurons in different layers.
    ///
    /// - **Task Management**: 
    ///   - Utilizes a `TaskManager` with multiple `TaskContainer<T>` instances for asynchronous task management.
    ///   - Efficiently manages task execution and synchronization.
    ///
    /// - **Initialization**: 
    ///   - Initializes sensory neurons, hidden layers, and motor (output) layers.
    ///   - Loads backpropagation and matrix-vector multiplication kernels for GPU execution.
    ///   - Initializes calculation tensors and task containers for neural network operations.
    ///
    /// - **GPU Acceleration**:
    ///   - Uses ILGPU for parallel processing of matrix-vector operations and backpropagation.
    ///   - Allocates and manages GPU buffers for efficient data transfer and computation.
    ///   - Custom kernels for backpropagation and matrix-vector multiplication.
    ///
    /// - **Training and Prediction**:
    ///   - Forward propagation method to calculate neuron activations.
    ///   - Backpropagation methods for updating weights and biases based on gradients.
    ///   - Training method to learn from input data over multiple epochs.
    ///   - Predict method for inference on new input data.
    ///
    /// - **Cost Function**:
    ///   - Supports Mean Squared Error (MSE) cost function.
    ///   - Calculates gradients for backpropagation using the derivative of the cost function.
    ///
    /// - **Utility Methods**:
    ///   - Methods for saving and loading the neural network state to and from files.
    ///   - Methods to handle changes in data and reinitialize necessary components.
    ///
    /// Aims:
    /// - **Efficiency**: Leveraging parallel processing and GPU acceleration to enhance the performance of neural network training and inference.
    /// - **Modularity**: Structured in a way that allows for easy extension and modification of the neural network architecture.
    /// - **Scalability**: Designed to handle large datasets and complex neural network structures by utilizing modern hardware capabilities.
    /// </summary>

    public class NeuralNetwork
    {
        private Parameters parameters;
        private readonly CostFunctionManager costFunctionManager;
        private readonly DataManager dataManager;
        private readonly AccelerationManager accelerationManager;
        private ITrainingManager trainingManager;

        #region ctor

        public NeuralNetwork(Parameters parameters)
        {
            dataManager = new DataManager();
            costFunctionManager = new CostFunctionManager(dataManager.Data, parameters);
            accelerationManager = new AccelerationManager(parameters);
            dataManager.InitializeData(parameters, parameters.SensoryNeurons);
            trainingManager = new BackPropergationManager(costFunctionManager, dataManager, accelerationManager, parameters, ForwardPropagate);
            this.parameters = parameters;
        }
        private NeuralNetwork(Matrix<INeuron> InData, Matrix<Neurite>[] InNeuriteTensor, Parameters parameters)
        {
            dataManager = new DataManager()
            {
                Data = InData,
                NeuriteTensor = InNeuriteTensor,
            };
            costFunctionManager = new CostFunctionManager(dataManager.Data, parameters);
            accelerationManager = new AccelerationManager(parameters);
            dataManager.InitializeData(parameters, parameters.SensoryNeurons, true);
            trainingManager = new BackPropergationManager(costFunctionManager, dataManager, accelerationManager, parameters, ForwardPropagate);
            this.parameters = parameters;
        }
        #pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
        private NeuralNetwork()
        #pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
        {

        }
        #endregion

        private void ForwardPropagate()
        {
            int columnCount = dataManager.Data.ColumnCount();
            for (int j = 1; j < columnCount; j++)
            {
                int previousLayerNeuronCount = dataManager.Neurons[j - 1].Length;
                int currentLayerNeuronCount = dataManager.Neurons[j].Length;

                // Initialize arrays
                double[] dotProduct = new double[currentLayerNeuronCount];

                Task<double[]> sourceNeuronsActivationsTask = Task.Run(() => dataManager.Neurons[j - 1].Where(r => r != null).Select(r => r.Activation).ToArray());

                Task<double[,]> neuritesWeightsTask = Task.Run(() => dataManager.NeuriteTensor[j - 1].SelectTranspose(r => r.Weight).ToArray());

                // Wait for all tasks to complete
                Task.WaitAll(sourceNeuronsActivationsTask, neuritesWeightsTask);

                dotProduct = accelerationManager.CalculateDotProduct(neuritesWeightsTask.Result, sourceNeuronsActivationsTask.Result);

                Parallel.For(0, dotProduct.Length, c =>
                {
                    dataManager.Neurons[j][c].Activation = dataManager.Neurons[j][c].activationFunction(dotProduct[c] + dataManager.Neurons[j][c].Bias);

                    if (double.IsNaN(dataManager.Neurons[j][c].Activation))
                    {
                        throw new InfinityException($"NaN detected in forward propagation at layer {j}, neuron {c}, type:{dataManager.Neurons[j][c].GetType().Name}");
                    }
                    if (double.IsInfinity(dataManager.Neurons[j][c].Activation))
                    {
                        throw new NaNException($"Infinity detected in forward propagation at layer {j}, neuron {c}, type:{dataManager.Neurons[j][c].GetType().Name}");
                    }
                });
            }
        }
        public void Train(double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false)
        {
            trainingManager.Train(trainingDataCollection, ExpectedResults, LearningRate, epochs, Silent);
        }
        public void Train(AccelerationType accelerationType, double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false)
        {
            trainingManager.Train(accelerationType, trainingDataCollection, ExpectedResults, LearningRate, epochs, Silent);
        }
        public double[] Predict(double[] inputValues)
        {
            trainingManager.SetSensoryNeuronsValues(inputValues);
            ForwardPropagate();

            INeuron[] outputNeurons = dataManager.Data.Column(dataManager.Data.ColumnCount() - 1);

            var prediction = outputNeurons.Select(neuron => neuron.Activation).ToArray();

            return outputNeurons.Select(neuron => neuron.Activation).ToArray();
        }

        #region IO
        public static NeuralNetwork Load(string FileName, AccelerationType accelerationType)
        {
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                Formatting = Formatting.Indented
            };

            settings.Converters.Add(new MatrixArrayJsonConverter<Neurite>());
            settings.Converters.Add(new MatrixJsonConverter<INeuron>());


            string json = File.ReadAllText(FileName);
            NeuralNetworkDto neuralNetworkDto = JsonConvert.DeserializeObject<NeuralNetworkDto>(json, settings) 
            ?? throw new InvalidOperationException("Deserialization failed, object is null.");

            Parameters parameters = new Parameters()
            {
                AccelerationType = accelerationType,
                SensoryNeurons = neuralNetworkDto.Data.Column(0).Select(r => (SensoryNeuron)r).ToArray(),
                NoHiddenLayers = neuralNetworkDto.NoHiddenLayers,
                HiddenLayerWidth = neuralNetworkDto.HiddenLayerWidth,
                NumberOfOutputs = neuralNetworkDto.NumberOfOutputs,
                DefaultActivationType = neuralNetworkDto.DefaultActivationType,
                CostFunction = neuralNetworkDto.costFunction,
                L2RegulationLamda = neuralNetworkDto.l2RegulationLamda,
                ClippingLimitUpper = neuralNetworkDto.clippingLimit_Upper,
                ClippingLimitLower = neuralNetworkDto.clippingLimit_Lower
            };
            if (parameters.CostFunction == CostFunctionType.huberLoss)
            {
                parameters.HuberLossDelta = neuralNetworkDto.HuberLossDelta;
            }
            NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkDto.Data, neuralNetworkDto.NeuriteTensor, parameters);

            return neuralNetwork;
        }
        public static NeuralNetwork TryLoad(string FileName, AccelerationType accelerationType)
        {
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                Formatting = Formatting.Indented
            };

            settings.Converters.Add(new MatrixArrayJsonConverter<Neurite>());
            settings.Converters.Add(new MatrixJsonConverter<INeuron>());


            string json = File.ReadAllText(FileName);
            NeuralNetworkDto neuralNetworkDto = JsonConvert.DeserializeObject<NeuralNetworkDto>(json, settings)
            ?? throw new InvalidOperationException("Deserialization failed, object is null.");

            Parameters parameters = new Parameters()
            {
                AccelerationType = accelerationType,
                SensoryNeurons = neuralNetworkDto.Data.Column(0).Select(r => (SensoryNeuron)r).ToArray(),
                NoHiddenLayers = neuralNetworkDto.NoHiddenLayers,
                HiddenLayerWidth = neuralNetworkDto.HiddenLayerWidth,
                NumberOfOutputs = neuralNetworkDto.NumberOfOutputs,
                DefaultActivationType = neuralNetworkDto.DefaultActivationType,
                CostFunction = neuralNetworkDto.costFunction,
                L2RegulationLamda = neuralNetworkDto.l2RegulationLamda,
                ClippingLimitUpper = neuralNetworkDto.clippingLimit_Upper,
                ClippingLimitLower = neuralNetworkDto.clippingLimit_Lower
            };
            if (parameters.CostFunction == CostFunctionType.huberLoss)
            {
                parameters.HuberLossDelta = neuralNetworkDto.HuberLossDelta;
            }
            NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkDto.Data, neuralNetworkDto.NeuriteTensor, parameters);

            return neuralNetwork;
        }
        public static void Save(NeuralNetwork neuralNetwork, string FileName)
        {
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                Formatting = Formatting.Indented
            };
            settings.Converters.Add(new MatrixArrayJsonConverter<Neurite>());
            settings.Converters.Add(new MatrixJsonConverter<INeuron>());


            var neuralNetworkDto = neuralNetwork.MapToDto();

            string json = JsonConvert.SerializeObject(neuralNetworkDto, settings);
            File.WriteAllText(FileName, json);
        }
        #endregion

    }
}