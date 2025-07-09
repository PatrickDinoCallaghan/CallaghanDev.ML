using CallaghanDev.ML.Exceptions;
using CallaghanDev.ML.Neural_Network;

namespace CallaghanDev.ML.NN.Training
{
    public class FFTrainingManager : TrainingManagerBase, ITrainingManager
    {

        public FFTrainingManager(CostFunctionManager costFunctionManager, DataManager dataManager,  IAccelerationManager accelerationManager, Parameters parameters): base(costFunctionManager, dataManager, accelerationManager, parameters) { }


        private double[] GetOutputLayerGradients(double[] costs, double learningRate)
        {
            int LastColumn = _dataManager.Data.ColumnCount() - 1;
            INeuron[] MotorNeurons = _dataManager.Data.Column(LastColumn).ToArray();
            int motorNeuronCount = MotorNeurons.Length;

            double[] gradients = new double[motorNeuronCount];

            Parallel.For(0, motorNeuronCount, i =>
            {
                INeuron motorNeuron = MotorNeurons[i];
                double activationDerivative = motorNeuron.activationFunctionDeriv(motorNeuron.Activation);
                gradients[i] = -costs[i] * activationDerivative;
            });

            double maxGradient = Math.Abs(gradients.Max());

            if (maxGradient > _parameters.GradientExplosionThreshold)
            {
                throw new TrainingFailureException(FailureType.ExplodingGradient);
            }
            return gradients;
        }
        private void UpdateOutputLayerGradients(double[] gradients, double learningRate, double lambda = 0)
        {

            int LastColumn = _dataManager.Data.ColumnCount() - 1;
            INeuron[] MotorNeurons = _dataManager.Data.Column(LastColumn).ToArray();
            int motorNeuronCount = MotorNeurons.Length;

            int dendriteCount = _dataManager.NeuriteTensor[LastColumn - 1].ColumnCount();

            Parallel.For(0, motorNeuronCount, i =>
            {
                INeuron motorNeuron = MotorNeurons[i];

                motorNeuron.Delta = gradients[i];

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

        private double[][] GetHiddenLayerGradients(double learningRate, double[] OutputDeltasArray)
        {
            int colCount = _parameters.NoHiddenLayers;
            List<double[]> doubles =new List<double[]>();
            for (int layerIndex = colCount; layerIndex > 0; layerIndex--)
            {
                INeuron[] precedingLayerNeurons = _dataManager.Neurons[layerIndex - 1];
                INeuron[] hiddenNeurons = _dataManager.Neurons[layerIndex];
                INeuron[] nextLayerNeurons = _dataManager.Neurons[layerIndex + 1];

                if (nextLayerNeurons.Length == 0 || hiddenNeurons.Length == 0 || precedingLayerNeurons.Length == 0)
                {
                    throw new IndexOutOfRangeException("Mismatch in neuron array sizes.");
                }

                int numCurrentNeurons = hiddenNeurons.Length;

                // Precompute required values
                double[] activationDerivatives = hiddenNeurons.Select(neuron => neuron.activationFunctionDeriv(neuron.Activation)).ToArray();

                // Arrays to store updated values
                double[] updatedDeltas = new double[numCurrentNeurons];

                for (int neuronIndex = 0; neuronIndex < numCurrentNeurons; neuronIndex++)
                {
                    // Calculate sum of weighted deltas for this neuron
                    double sumOfWeightedDeltas = 0.0;
                    for (int i = 0; i < nextLayerNeurons.Length; i++)
                    {
                        sumOfWeightedDeltas += _dataManager.NeuriteTensor[layerIndex][i, neuronIndex].Weight * OutputDeltasArray[i];
                    }

                    // Calculate the delta for the current neuron
                    updatedDeltas[neuronIndex] = sumOfWeightedDeltas * activationDerivatives[neuronIndex];
                }
                OutputDeltasArray = updatedDeltas;
                doubles.Add(updatedDeltas);
            }
            return doubles.ToArray();
        }

        private void UpdateHiddenLayerGradients(double learningRate, double[][] deltas)
        {
            int colCount = _parameters.NoHiddenLayers;

            for (int layerIndex = colCount; layerIndex > 0; layerIndex--)
            {
                INeuron[] precedingLayerNeurons = _dataManager.Neurons[layerIndex - 1];
                INeuron[] hiddenNeurons = _dataManager.Neurons[layerIndex];

                if (hiddenNeurons.Length == 0 || precedingLayerNeurons.Length == 0)
                {
                    throw new IndexOutOfRangeException("Mismatch in neuron array sizes.");
                }

                int numCurrentNeurons = hiddenNeurons.Length;
                int numPrecedingNeurons = precedingLayerNeurons.Length;

                double[] precedingLayerActivations = precedingLayerNeurons.Select(neuron => neuron.Activation).ToArray();
                double[] biases = hiddenNeurons.Select(neuron => neuron.Bias).ToArray();

                // Apply the updated values back to the neurons
                for (int i = 0; i < numCurrentNeurons; i++)
                {
                    hiddenNeurons[i].Delta = deltas[layerIndex - 1][i];
                    for (int k = 0; k < numPrecedingNeurons; k++)
                    {
                        _dataManager.NeuriteTensor[layerIndex - 1][i, k].Weight -= learningRate * deltas[layerIndex - 1][i] * precedingLayerActivations[k];
                    }
                    hiddenNeurons[i].Bias = biases[i] - learningRate * deltas[layerIndex - 1][i];
                }

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
                        Terminal.Gui.Helpers.ProgressBarHelper.DisplayProgressBar(Counter, MaxVal, "Progress");
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

        private void ForwardPropagate()
        {
            int columnCount = _dataManager.Data.ColumnCount();
            for (int j = 1; j < columnCount; j++)
            {
                int previousLayerNeuronCount = _dataManager.Neurons[j - 1].Length;
                int currentLayerNeuronCount = _dataManager.Neurons[j].Length;

                // Initialize arrays
                double[] dotProduct = new double[currentLayerNeuronCount];

                Task<double[]> sourceNeuronsActivationsTask = Task.Run(() => _dataManager.Neurons[j - 1].Where(r => r != null).Select(r => r.Activation).ToArray());

                Task<double[,]> neuritesWeightsTask = Task.Run(() => _dataManager.NeuriteTensor[j - 1].SelectTranspose(r => r.Weight).ToArray());

                // Wait for all tasks to complete
                Task.WaitAll(sourceNeuronsActivationsTask, neuritesWeightsTask);

                dotProduct = _accelerationManager.CalculateDotProduct(neuritesWeightsTask.Result, sourceNeuronsActivationsTask.Result);

                Parallel.For(0, dotProduct.Length, c =>
                {
                    _dataManager.Neurons[j][c].Activation = _dataManager.Neurons[j][c].activationFunction(dotProduct[c] + _dataManager.Neurons[j][c].Bias);

                    if (double.IsNaN(_dataManager.Neurons[j][c].Activation))
                    {
                        throw new InfinityException($"NaN detected in forward propagation at layer {j}, neuron {c}, type:{_dataManager.Neurons[j][c].GetType().Name}");
                    }
                    if (double.IsInfinity(_dataManager.Neurons[j][c].Activation))
                    {
                        throw new NaNException($"Infinity detected in forward propagation at layer {j}, neuron {c}, type:{_dataManager.Neurons[j][c].GetType().Name}");
                    }
                });
            }
        }

        private void Learn(double[] trainingData, double[] ExpectedResult, double LearningRate)
        {
            SetSensoryNeuronsValues(trainingData);
            ForwardPropagate();
            BackPropagate(LearningRate, ExpectedResult);
        }
        private void BackPropagate(double learningRate, double[] expectedOutputValues)
        {
            double[] costs = _costFunctionManager.CalculateCost(expectedOutputValues);

            double[] OutputGradients = GetOutputLayerGradients(costs, learningRate);
            double[][] HiddenGradients = GetHiddenLayerGradients(learningRate, OutputGradients);

            double Scale = GetGlobalGradientClippingScale(OutputGradients, HiddenGradients);

            OutputGradients = OutputGradients.Select(g => g *= Scale).ToArray();
            HiddenGradients = HiddenGradients.Select(ar => ar.Select(g=>g*=Scale).ToArray()).ToArray();

            double combinedNorm = Math.Sqrt(
            OutputGradients.Select(g => g * g).Sum() +
            HiddenGradients.SelectMany(g => g).Select(g => g * g).Sum());

            if (combinedNorm > _parameters.GradientExplosionThreshold )
            {
                throw new TrainingFailureException(FailureType.ExplodingGradient);
            }

            if (combinedNorm < _parameters.GradientVanishingThreshold && !expectedOutputValues.Any(r=>r==0))
            {
                throw new TrainingFailureException(FailureType.VanishingGradient);
            }

            UpdateOutputLayerGradients(OutputGradients, learningRate, _parameters.L2RegulationLamda);
            UpdateHiddenLayerGradients(learningRate, HiddenGradients);
        }

        private double GetGlobalGradientClippingScale(double[] OutputGradients, double[][] HiddenGradients)
        {
            double globalNormSquared = OutputGradients.Select(g => g * g).Sum();
            foreach (var layerGradients in HiddenGradients)
            {
                globalNormSquared += layerGradients.Select(g => g * g).Sum();
            }
            double globalNorm = Math.Sqrt(globalNormSquared);

            // Determine scaling factor
            double scale = 1.0;
            if (globalNorm > _parameters.GradientClippingThreshold)
            {
                scale = _parameters.GradientClippingThreshold / globalNorm;
            }

            return scale;
        }

        public void ComputeOutputs() => ForwardPropagate();

        public void UpdateParameters(double learningRate, double[] expectedOutputValues) => BackPropagate(learningRate, expectedOutputValues);
    }

}
