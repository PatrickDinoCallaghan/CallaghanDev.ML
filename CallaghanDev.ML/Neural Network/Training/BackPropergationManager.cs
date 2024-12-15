using CallaghanDev.ML.Neural_Network;
using CallaghanDev.ML.Neural_Network.Exceptions;
using CallaghanDev.Utilities.ConsoleHelper;
using MathNet.Numerics.Statistics;
using System.Diagnostics.Metrics;

namespace CallaghanDev.ML.NN.Training
{
    public class BackPropergationManager : TrainingManagerBase, ITrainingManager
    {
        private List<BatchNormalization> batchNormalizationLayers;

        public BackPropergationManager(
            CostFunctionManager costFunctionManager,
            DataManager dataManager,
            AccelerationManager accelerationManager,
            Parameters parameters,
            Action ForwardPropagate)
            : base(costFunctionManager, dataManager, accelerationManager, parameters, ForwardPropagate)
        {
            // Initialize BatchNormalization instances for each layer
            batchNormalizationLayers = new List<BatchNormalization>();
            for (int i = 0; i < parameters.NoHiddenLayers; i++)
            {
                int numFeatures = dataManager.Neurons[i].Length; // Number of neurons in the current layer
                batchNormalizationLayers.Add(new BatchNormalization(numFeatures));
            }
        }

        private void UpdateOutputLayerGradients(double[] costs, double learningRate, double lambda = 0)
        {
            int LastColumn = _dataManager.Data.ColumnCount() - 1;
            INeuron[] MotorNeurons = _dataManager.Data.Column(LastColumn).ToArray();
            int motorNeuronCount = MotorNeurons.Length;

            double[,] dendritesWeights = GetWeightsMatrix(LastColumn - 1);

            int dendriteCount = dendritesWeights.GetUpperBound(1) + 1;

            double[] gradients = new double[motorNeuronCount];

            Parallel.For(0, motorNeuronCount, i =>
            {
                INeuron motorNeuron = MotorNeurons[i];
                double activationDerivative = motorNeuron.activationFunctionDeriv(motorNeuron.Activation);
                gradients[i] = -costs[i] * activationDerivative;
            });

            gradients = ClipGradients(gradients, _parameters.GradientClippingThreshold);

            Parallel.For(0, motorNeuronCount, i =>
            {
                INeuron motorNeuron = MotorNeurons[i];

                motorNeuron.Delta = gradients[i];
                int dendriteCount = dendritesWeights.GetUpperBound(1) + 1;

                for (int j = 0; j < dendriteCount; j++)
                {
                    double weightUpdate = learningRate * (gradients[i] * _dataManager.Neurons[LastColumn - 1][j].Activation + lambda * _dataManager.NeuriteTensor[LastColumn - 1][i, j].Weight);

                    _dataManager.NeuriteTensor[LastColumn - 1][i, j].Weight -= weightUpdate;

                    if (double.IsNaN(_dataManager.NeuriteTensor[LastColumn - 1][i, j].Weight))
                    {
                        throw new NaNException();
                    }
                    if (double.IsInfinity(_dataManager.NeuriteTensor[LastColumn - 1][i, j].Weight))
                    {
                        throw new InfinityException();
                    }
                }

                motorNeuron.Bias -= learningRate * gradients[i];
            });
        }

        private void UpdateHiddenLayerGradients(double learningRate)
        {
            int colCount = _parameters.NoHiddenLayers;

            for (int layerIndex = colCount; layerIndex > 0; layerIndex--)
            {
                INeuron[] precedingLayerNeurons = _dataManager.Neurons[layerIndex - 1];
                INeuron[] hiddenNeurons = _dataManager.Neurons[layerIndex];
                INeuron[] nextLayerNeurons = _dataManager.Neurons[layerIndex + 1];

                if (nextLayerNeurons.Length == 0 || hiddenNeurons.Length == 0 || precedingLayerNeurons.Length == 0)
                {
                    throw new IndexOutOfRangeException("Mismatch in neuron array sizes.");
                }

                double[,] weightsMatrix = GetWeightsMatrix(layerIndex);
                int numCurrentNeurons = hiddenNeurons.Length;
                int numPrecedingNeurons = precedingLayerNeurons.Length;

                Task<double[]> taskDeltasArray = Task.Run(() => nextLayerNeurons.Select(neuron => neuron.Delta).ToArray());
                Task<double[]> taskPrecedingLayerNeuronsActivations = Task.Run(() => precedingLayerNeurons.Select(r => r.Activation).ToArray());
                Task<double[]> taskHiddenNeuronsActivationFunctionDeriv = Task.Run(() => hiddenNeurons.Select(r => r.activationFunctionDeriv(r.Activation)).ToArray());
                Task<double[]> taskHiddenNeuronsBiases = Task.Run(() => hiddenNeurons.Select(r => r.Bias).ToArray());

                Task.WaitAll(taskDeltasArray, taskPrecedingLayerNeuronsActivations, taskHiddenNeuronsActivationFunctionDeriv, taskHiddenNeuronsBiases);

                double[] deltasArray = taskDeltasArray.Result;
                double[] precedingLayerNeuronsActivations = taskPrecedingLayerNeuronsActivations.Result;
                double[] hiddenNeurons_activationFunctionDeriv = taskHiddenNeuronsActivationFunctionDeriv.Result;
                double[] hiddenNeurons_Biases = taskHiddenNeuronsBiases.Result;

                var GPUBackpropResult = _accelerationManager.CalculateBackpropagationValues(
                    numCurrentNeurons,
                    numPrecedingNeurons,
                    weightsMatrix,
                    deltasArray,
                    precedingLayerNeuronsActivations,
                    hiddenNeurons_activationFunctionDeriv,
                    hiddenNeurons_Biases,
                    learningRate,
                    _parameters.GradientClippingThreshold);

                Parallel.For(0, numCurrentNeurons, i =>
                {
                    hiddenNeurons[i].Delta = GPUBackpropResult.updatedDeltas[i];
                    for (int k = 0; k < numPrecedingNeurons; k++)
                    {
                        _dataManager.NeuriteTensor[layerIndex - 1][i, k].Weight -= GPUBackpropResult.updatedWeights[k, i];
                    }
                    hiddenNeurons[i].Bias = GPUBackpropResult.updatedBiases[i];
                });
            }
        }

        public override void Train(double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false)
        {
            long MaxVal = (long)epochs * (long)trainingDataCollection.Length;
            long Counter = 0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int i = 0; i < trainingDataCollection.Length; i++)
                {
                    Learn(trainingDataCollection[i], ExpectedResults[i], LearningRate);
                    Counter++;
                    if (!Silent)
                    {
                        cnsl.DisplayProgressBar(Counter, MaxVal, "Progress");
                    }
                }
                OnEpochFinished?.Invoke(epoch);
            }
        }
        public override void Train(AccelerationType accelerationType, double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false)
        {
            _parameters.AccelerationType = accelerationType;
            Train(trainingDataCollection, ExpectedResults, LearningRate, epochs, Silent);
        }

        private void Learn(double[] trainingData, double[] ExpectedResult, double LearningRate)
        {
            SetSensoryNeuronsValues(trainingData);

            // Apply batch normalization before propagating activations
            for (int layerIndex = 0; layerIndex < batchNormalizationLayers.Count; layerIndex++)
            {
                double[] activations = _dataManager.Neurons[layerIndex].Select(n => n.Activation).ToArray();
                double[] normalizedActivations = batchNormalizationLayers[layerIndex].Forward(activations, isTraining: true);

                // Update neuron activations
                for (int i = 0; i < normalizedActivations.Length; i++)
                {
                    _dataManager.Neurons[layerIndex][i].Activation = normalizedActivations[i];
                }
            }

            _ForwardPropagate();
            BackPropagate(LearningRate, ExpectedResult);
        }
        private void BackPropagate(double learningRate, double[] expectedOutputValues)
        {
            double[] costs = _costFunctionManager.CalculateCost(expectedOutputValues);

            UpdateOutputLayerGradients(costs, learningRate, _parameters.L2RegulationLamda);
            UpdateHiddenLayerGradients(learningRate);
        }
        private double[,] GetWeightsMatrix(int layerIndex)
        {

            double[,] Weights = new double[_dataManager.NeuriteTensor[layerIndex].RowCount(), _dataManager.NeuriteTensor[layerIndex].ColumnCount()];// NeuriteTensor[layerIndex].Select(r => r.Weight).ToArray();

            int rowCount = _dataManager.NeuriteTensor[layerIndex].RowCount();
            int colCount = _dataManager.NeuriteTensor[layerIndex].ColumnCount();


            if (rowCount > colCount)
            {
                Parallel.For(0, rowCount, i =>
                {
                    for (int j = 0; j < colCount; j++)
                    {
                        Weights[i, j] = _dataManager.NeuriteTensor[layerIndex][i, j].Weight;
                    }
                });
            }
            else
            {
                Parallel.For(0, colCount, j =>
                {
                    for (int i = 0; i < rowCount; i++)
                    {
                        Weights[i, j] = _dataManager.NeuriteTensor[layerIndex][i, j].Weight;
                    }
                });
            }
            return Weights;
        }

        double[] ClipGradients(double[] gradients, double threshold)
        {
            double norm = Math.Sqrt(gradients.Sum(g => g * g));
            if (norm > threshold)
            {
                for (int i = 0; i < gradients.Length; i++)
                {
                    gradients[i] *= threshold / norm;
                }
            }
            return gradients;
        }
    }
    /*
    public class BackPropergationManager : TrainingManagerBase, ITrainingManager
    {
        BatchNormalization batchNormalization;
        public BackPropergationManager(CostFunctionManager costFunctionManager, DataManager dataManager, AccelerationManager accelerationManager, Parameters parameters, Action ForwardPropagate) : base(costFunctionManager, dataManager, accelerationManager, parameters, ForwardPropagate) 
        {
           
        }
        private void UpdateOutputLayerGradients(double[] costs, double learningRate, double lambda = 0)
        {
            int LastColumn = _dataManager.Data.ColumnCount() - 1;
            INeuron[] MotorNeurons = _dataManager.Data.Column(LastColumn).ToArray();
            int motorNeuronCount = MotorNeurons.Length;


            double[,] dendritesWeights = GetWeightsMatrix(LastColumn - 1);

            Parallel.For(0, motorNeuronCount, i =>
            {
                INeuron motorNeuron = MotorNeurons[i];
                double activationDerivative = motorNeuron.activationFunctionDeriv(motorNeuron.Activation);
                double gradient = -costs[i] * activationDerivative;
                gradient = Math.Max(_parameters.ClippingLimitLower, Math.Min(_parameters.ClippingLimitUpper, gradient));
                motorNeuron.Delta = gradient;
                int dendriteCount = dendritesWeights.GetUpperBound(1) + 1;


                for (int j = 0; j < dendriteCount; j++)
                {
                    double weightUpdate = learningRate * (gradient * _dataManager.Neurons[LastColumn - 1][j].Activation + lambda * _dataManager.NeuriteTensor[LastColumn - 1][i, j].Weight);

                    _dataManager.NeuriteTensor[LastColumn - 1][i, j].Weight -= weightUpdate;

                    if (double.IsNaN(_dataManager.NeuriteTensor[LastColumn - 1][i, j].Weight))
                    {
                        throw new NaNException();
                    }
                    if (double.IsInfinity(_dataManager.NeuriteTensor[LastColumn - 1][i, j].Weight))
                    {
                        throw new InfinityException();
                    }
                }

                motorNeuron.Bias -= learningRate * gradient;
            });
        }

        // Utility function to extract the weight matrix of a specific layer
        private double[,] GetWeightsMatrix(int layerIndex)
        {

            double[,] Weights = new double[_dataManager.NeuriteTensor[layerIndex].RowCount(), _dataManager.NeuriteTensor[layerIndex].ColumnCount()];// NeuriteTensor[layerIndex].Select(r => r.Weight).ToArray();

            int rowCount = _dataManager.NeuriteTensor[layerIndex].RowCount();
            int colCount = _dataManager.NeuriteTensor[layerIndex].ColumnCount();


            if (rowCount > colCount)
            {
                Parallel.For(0, rowCount, i =>
                {
                    for (int j = 0; j < colCount; j++)
                    {
                        Weights[i, j] = _dataManager.NeuriteTensor[layerIndex][i, j].Weight;
                    }
                });
            }
            else
            {
                Parallel.For(0, colCount, j =>
                {
                    for (int i = 0; i < rowCount; i++)
                    {
                        Weights[i, j] = _dataManager.NeuriteTensor[layerIndex][i, j].Weight;
                    }
                });
            }
            return Weights;
        }

        private void UpdateHiddenLayerGradients(double learningRate)
        {
            int colCount = _parameters.NoHiddenLayers;

            for (int layerIndex = colCount; layerIndex > 0; layerIndex--)
            {
                INeuron[] precedingLayerNeurons = _dataManager.Neurons[layerIndex - 1];

                INeuron[] hiddenNeurons = _dataManager.Neurons[layerIndex];

                INeuron[] nextLayerNeurons = _dataManager.Neurons[layerIndex + 1];


                if (nextLayerNeurons.Length == 0 || hiddenNeurons.Length == 0 || precedingLayerNeurons.Length == 0)
                {
                    throw new IndexOutOfRangeException("Mismatch in neuron array sizes.");
                }

                double[,] weightsMatrix = GetWeightsMatrix(layerIndex);

                int numCurrentNeurons = hiddenNeurons.Length;
                int numPrecedingNeurons = precedingLayerNeurons.Length;

                Task<double[]> taskDeltasArray = Task.Run(() => nextLayerNeurons.Select(neuron => neuron.Delta).ToArray());
                Task<double[]> taskPrecedingLayerNeuronsActivations = Task.Run(() => precedingLayerNeurons.Select(r => r.Activation).ToArray());
                Task<double[]> taskHiddenNeuronsActivationFunctionDeriv = Task.Run(() => hiddenNeurons.Select(r => r.activationFunctionDeriv(r.Activation)).ToArray());
                Task<double[]> taskHiddenNeuronsBiases = Task.Run(() => hiddenNeurons.Select(r => r.Bias).ToArray());

                // Wait for all tasks to complete
                Task.WaitAll(taskDeltasArray, taskPrecedingLayerNeuronsActivations, taskHiddenNeuronsActivationFunctionDeriv, taskHiddenNeuronsBiases);

                // Retrieve results
                double[] deltasArray = taskDeltasArray.Result;
                double[] precedingLayerNeuronsActivations = taskPrecedingLayerNeuronsActivations.Result;
                double[] hiddenNeurons_activationFunctionDeriv = taskHiddenNeuronsActivationFunctionDeriv.Result;
                double[] hiddenNeurons_Biases = taskHiddenNeuronsBiases.Result;


                var GPUBackpropResult = _accelerationManager.CalculateBackpropagationValues(
                    numCurrentNeurons,
                    numPrecedingNeurons,
                    weightsMatrix,
                    deltasArray,
                    precedingLayerNeuronsActivations,
                    hiddenNeurons_activationFunctionDeriv,
                    hiddenNeurons_Biases,
                    learningRate,
                    _parameters.ClippingLimitLower,
                    _parameters.ClippingLimitUpper);

                Parallel.For(0, numCurrentNeurons, i =>
                {
                    hiddenNeurons[i].Delta = GPUBackpropResult.updatedDeltas[i];

                    for (int k = 0; k < numPrecedingNeurons; k++)
                    {
                        _dataManager.NeuriteTensor[layerIndex - 1][i, k].Weight -= GPUBackpropResult.updatedWeights[k, i]; // CopyToCPU transposes the data into column major order so the result will have k,i instead of i,k
                    }
                    hiddenNeurons[i].Bias = GPUBackpropResult.updatedBiases[i];
                });


            }
        }

        public override void Train(double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false)
        {
            long MaxVal = (long)epochs * (long)trainingDataCollection.Length;
            long Counter = 0;
            bool First = true;
            for (int e = 0; e < epochs; e++)
            {
                for (int i = 0; i < trainingDataCollection.Length; i++)
                {
                    if (!Silent && First)
                    {
                        cnsl.DisplayProgressBar(Counter, MaxVal, "Progress");
                        First = false;
                    }
                    Learn(trainingDataCollection[i], ExpectedResults[i], LearningRate);
                    Counter++;
                    if (!Silent & !First)
                    {
                        cnsl.DisplayProgressBar(Counter, MaxVal, "Progress");
                    }
                }
                OnEpochFinished?.Invoke(e);
            }
        }
        public override void Train(AccelerationType accelerationType, double[][] trainingDataCollection, double[][] ExpectedResults, double LearningRate, int epochs, bool Silent = false)
        {
            _parameters.AccelerationType = accelerationType;
            Train(trainingDataCollection, ExpectedResults, LearningRate, epochs, Silent);
        }

        private void Learn(double[] trainingData, double[] ExpectedResult, double LearningRate)
        {
            SetSensoryNeuronsValues(trainingData);
            _ForwardPropagate();
            BackPropagate(LearningRate, ExpectedResult);
        }
        private void BackPropagate(double learningRate, double[] expectedOutputValues)
        {
            double[] costs = _costFunctionManager.CalculateCost(expectedOutputValues);

            UpdateOutputLayerGradients(costs, learningRate, _parameters.L2RegulationLamda);
            UpdateHiddenLayerGradients(learningRate);
        }


    }*/
    public class BatchNormalization
    {
        private double[] gamma;  // Scale parameter
        private double[] beta;   // Shift parameter
        private double[] runningMean;
        private double[] runningVariance;
        private double epsilon;
        private double momentum;

        public BatchNormalization(int numFeatures, double epsilon = 1e-5, double momentum = 0.9)
        {
            gamma = Enumerable.Repeat(1.0, numFeatures).ToArray();  // Initialize to 1
            beta = new double[numFeatures];                        // Initialize to 0
            runningMean = new double[numFeatures];
            runningVariance = new double[numFeatures];
            this.epsilon = epsilon;
            this.momentum = momentum;
        }

        public double[] Forward(double[] inputs, bool isTraining)
        {
            int numFeatures = inputs.Length; // The number of features in the input
            double[] normalized = new double[numFeatures];
            double[] mean = new double[numFeatures];
            double[] variance = new double[numFeatures];

            if (isTraining)
            {
                // Compute batch mean
                double batchMean = inputs.Average();
                for (int i = 0; i < numFeatures; i++)
                {
                    mean[i] = batchMean;
                }

                // Compute batch variance
                double batchVariance = inputs.Select(x => Math.Pow(x - batchMean, 2)).Average();
                for (int i = 0; i < numFeatures; i++)
                {
                    variance[i] = batchVariance;
                }

                // Update running mean and variance
                for (int i = 0; i < numFeatures; i++)
                {
                    runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
                    runningVariance[i] = momentum * runningVariance[i] + (1 - momentum) * variance[i];
                }
            }
            else
            {
                // Use running mean and variance during inference
                mean = runningMean;
                variance = runningVariance;
            }

            // Normalize inputs
            for (int i = 0; i < numFeatures; i++)
            {
                normalized[i] = (inputs[i] - mean[i]) / Math.Sqrt(variance[i] + epsilon);
            }

            // Apply scale (gamma) and shift (beta)
            for (int i = 0; i < numFeatures; i++)
            {
                normalized[i] = gamma[i] * normalized[i] + beta[i];
            }

            return normalized;
        }

        public void Backward(double[] inputs, double[] gradients)
        {
            int batchSize = inputs.Length;

            // Compute gradients for gamma and beta
            double[] dGamma = new double[gamma.Length];
            double[] dBeta = new double[beta.Length];

            for (int i = 0; i < inputs.Length; i++)
            {
                dGamma[i] += gradients[i] * (inputs[i] - runningMean[i]) / Math.Sqrt(runningVariance[i] + epsilon);
                dBeta[i] += gradients[i];
            }

            // Update gamma and beta
            for (int i = 0; i < gamma.Length; i++)
            {
                gamma[i] -= dGamma[i];
                beta[i] -= dBeta[i];
            }
        }
    }

}
