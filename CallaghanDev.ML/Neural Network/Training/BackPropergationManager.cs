using CallaghanDev.ML.Neural_Network;
using CallaghanDev.ML.Neural_Network.Exceptions;
using CallaghanDev.Utilities.ConsoleHelper;

namespace CallaghanDev.ML.NN.Training
{
    public class BackPropergationManager : TrainingManagerBase, ITrainingManager
    {
        public BackPropergationManager(CostFunctionManager costFunctionManager, DataManager dataManager, AccelerationManager accelerationManager, Parameters parameters, Action ForwardPropagate) : base(costFunctionManager, dataManager, accelerationManager, parameters, ForwardPropagate) { }

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
            for (int e = 0; e < epochs; e++)
            {
                for (int i = 0; i < trainingDataCollection.Length; i++)
                {
                    if (!Silent)
                    {
                        cnsl.DisplayProgressBar(Counter, MaxVal, "Progress");
                    }
                    Learn(trainingDataCollection[i], ExpectedResults[i], LearningRate);
                    Counter++;
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


    }
}
