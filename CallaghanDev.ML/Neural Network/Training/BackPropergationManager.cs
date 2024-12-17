using CallaghanDev.ML.Neural_Network;
using CallaghanDev.ML.Neural_Network.Exceptions;
using CallaghanDev.Utilities.ConsoleHelper;
using MathNet.Numerics.Statistics;
using System.Diagnostics.Metrics;

namespace CallaghanDev.ML.NN.Training
{
    public class BackPropergationManager : TrainingManagerBase, ITrainingManager
    {

        public BackPropergationManager(
            CostFunctionManager costFunctionManager,
            DataManager dataManager,
            AccelerationManager accelerationManager,
            Parameters parameters,
            Action ForwardPropagate)
            : base(costFunctionManager, dataManager, accelerationManager, parameters, ForwardPropagate) { }

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

            _ForwardPropagate.Invoke();
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

}
