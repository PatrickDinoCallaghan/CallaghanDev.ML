using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;

namespace CallaghanDev.ML
{
    public class NeuralNetwork
    {
        private Data data;
        private IAccelerationManager accelerationManager;

        private CostFunction CostFunctionDeriv { get; set; }
        private delegate double CostFunction(double value, double prediction);
        private CostFunction costFunction { get; set; }

        public NeuralNetwork(Parameters parameters)
        {
            data = new Data(parameters);
            InitAcceleration();
            InitCostFunction();
        }
        private NeuralNetwork(Data Indata)
        {
            data = Indata;
            InitAcceleration();
            InitCostFunction();
        }
        private void InitAcceleration()
        {

            if (data.parameters.AccelerationType == AccelerationType.GPU)
            {
                accelerationManager = new AccelerationGPU();
            }
            else if (data.parameters.AccelerationType == AccelerationType.CPU)
            {
                accelerationManager = new AccelerationCPU();
            }
            else
            {
                throw new Exception("Unsupported AccelerationType. Supported types are CPU and GPU.");
            }
        }

        public void Train(double[][] inputs, double[][] expected, double LearningRate, int epochs, bool silent = false)
        {
            long total = (long)inputs.Length * epochs;
            long count = 0;

            for (int e = 0; e < epochs; e++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    Learn(inputs[i], expected[i], LearningRate);
                    count++;
                    if (!silent)
                    {
                        Terminal.Gui.Helpers.ProgressBarHelper.DisplayProgressBar(count, total, "Progress");
                    }
                }
            }
        }

        private void Learn(double[] x, double[] y, double LearningRate)
        {
            SetInputLayer(x);
            ForwardPropagate();

            var last = data.layers[^1];
            double[] costDerivs = new double[last.Size];
            for (int i = 0; i < last.Size; i++)
            {
                costDerivs[i] = CostFunctionDeriv(last.Activations[i], y[i]);
            }
            BackPropagate(costDerivs, LearningRate);
        }

        private void SetInputLayer(double[] x)
        {
            Array.Copy(x, data.layers[0].Activations, x.Length);
        }

        private void ForwardPropagate()
        {
            for (int l = 1; l < data.layers.Length; l++)
            {
                var prev = data.layers[l - 1];
                var cur = data.layers[l];

                var dot = accelerationManager.CalculateDotProduct(cur.Weights, prev.Activations);

                var (act, der) = accelerationManager.ActivateLayer(dot, cur.Biases, cur.ActivationType);

                cur.Activations = act;
                cur.Derivatives = der;
            }
        }

        private void BackPropagate(double[] costDerivs, double LearningRate)
        {
            int L = data.layers.Length - 1;

            // output layer
            var outLayer = data.layers[L];
            var outDeltas = accelerationManager.CalculateOutputGradients(costDerivs, outLayer.Derivatives);

            outLayer.Weights = accelerationManager.UpdateWeights(
                outLayer.Weights,
                outDeltas,
                data.layers[L - 1].Activations,
                LearningRate
            );
            outLayer.Biases = accelerationManager.UpdateBias(outLayer.Biases, outDeltas, LearningRate);

            // hidden layers
            var nextDeltas = outDeltas;
            for (int l = L - 1; l >= 1; l--)
            {
                var layer = data.layers[l];
                var weights = data.layers[l + 1].Weights;

                var hidDeltas = accelerationManager.CalculateHiddenGradients(
                    weights,
                    nextDeltas,
                    layer.Derivatives
                );

                layer.Weights = accelerationManager.UpdateWeights(layer.Weights, hidDeltas, data.layers[l - 1].Activations, LearningRate);

                layer.Biases = accelerationManager.UpdateBias(layer.Biases, hidDeltas, LearningRate);

                nextDeltas = hidDeltas;
            }
        }
        public void SetSensoryNeuronsValues(double[] inputValues)
        {
            var inputLayer = data.layers[0].Activations;
            Parallel.For(0, inputValues.Length, i =>
                inputLayer[i] = inputValues[i]
            );
        }
        public double[] Predict(double[] inputValues)
        {
            SetSensoryNeuronsValues(inputValues);
            ForwardPropagate();
            return data.layers[^1].Activations.ToArray();
        }
        public void ComputeOutputs() => ForwardPropagate();


        private void InitCostFunction()
        {
            switch (data.parameters.CostFunction)
            {
                case CostFunctionType.mse:
                    costFunction = (double ActualValue, double predicted) => { return Math.Pow(predicted - ActualValue, 2); };
                    CostFunctionDeriv = (double ActualValue, double predicted) => { return 2 * (predicted - ActualValue); };
                    break;
                case CostFunctionType.ZeroWeightedMSE:
                    costFunction = (double ActualValue, double predicted) => {
                        double zeroWeight = 0.25;
                        double nonZeroWeight = 1.0;

                        // Determine the weight based on the true value
                        double weight = (ActualValue == 0) ? zeroWeight : nonZeroWeight;

                        // Calculate the weighted MSE for the single prediction
                        return weight * Math.Pow(predicted - ActualValue, 2);
                    };
                    CostFunctionDeriv = (double ActualValue, double predicted) => {
                        double zeroWeight = 0.25;
                        double nonZeroWeight = 1.0;

                        // Determine the weight based on the true value
                        double weight = (ActualValue == 0) ? zeroWeight : nonZeroWeight;

                        // Calculate the derivative of the weighted MSE for the single prediction
                        return 2 * weight * (predicted - ActualValue);
                    };
                    break;
                case CostFunctionType.binaryCrossEntropy:
                    costFunction = (double ActualValue, double predicted) => { return -(ActualValue * Math.Log(predicted) + (1 - ActualValue) * Math.Log(1 - predicted)); };
                    CostFunctionDeriv = (double ActualValue, double predicted) => { return (predicted - ActualValue) / (predicted * (1 - predicted)); };
                    break;
                case CostFunctionType.mae:
                    costFunction = (double ActualValue, double predicted) => { return Math.Abs(ActualValue - predicted); };
                    CostFunctionDeriv = (double ActualValue, double predicted) => { return ActualValue > predicted ? -1 : 1; };
                    break;
                case CostFunctionType.huberLoss:
                    costFunction = (ActualValue, predicted) => {
                        double diff = ActualValue - predicted;
                        if (Math.Abs(diff) <= data.parameters.HuberLossDelta)
                        {
                            return 0.5 * diff * diff;
                        }
                        else
                        {
                            return data.parameters.HuberLossDelta * (Math.Abs(diff) - 0.5 * data.parameters.HuberLossDelta);
                        }};
                    CostFunctionDeriv = (ActualValue, predicted) => {
                        double diff = ActualValue - predicted;
                        if (Math.Abs(diff) <= data.parameters.HuberLossDelta)
                        {
                            return diff;
                        }
                        else
                        {
                            return data.parameters.HuberLossDelta * Math.Sign(diff);
                        }
                    };
                    break;
                case CostFunctionType.categoricalCrossEntropy:
                    costFunction = (ActualValue, predicted) => {
                        return -ActualValue * Math.Log(predicted + 1e-15);
                    };
                    CostFunctionDeriv = (double ActualValue, double predicted) => { return predicted - ActualValue; };
                    break;
                default:
                    throw new ArgumentException("Unsupported cost function type");
            }
        }
        public double[] CalculateCost(double[] expectedOutputValues)
        {
            int columnIndex = data.layers.Length - 1;

            double[] costDifferences = new double[data.layers[columnIndex].Size];

            Parallel.For(0, data.layers[columnIndex].Size, i =>
            {
                double nnOutputValue = data.layers[columnIndex].Activations[i];
                costDifferences[i] = CostFunctionDeriv(nnOutputValue, expectedOutputValues[i]);
            });

            return costDifferences;
        }

        public static NeuralNetwork Load(string FileName, AccelerationType accelerationType)
        {
            Data data = Data.Load(FileName);
            NeuralNetwork nn = new NeuralNetwork(data);
            data.parameters.AccelerationType = accelerationType;
            nn.data = data;
            return nn;
        }
        public void Save(string FileName)
        {
            data.Save(FileName);
        }
    }
}
