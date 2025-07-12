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

            if (data.parameters.AccelerationType == AccelerationType.GPU || data.parameters.AccelerationType == AccelerationType.CUDA)
            {
                accelerationManager = new AccelerationGPU(data.parameters.AccelerationType , data.parameters.AccelerationDeviceId);
            }
            else if (data.parameters.AccelerationType == AccelerationType.CPU)
            {
                accelerationManager = new AccelerationCPU();
            }
            else if (data.parameters.AccelerationType == AccelerationType.MutliThreadCPU)
            {
                accelerationManager = new AccelerationMutliThreadCPU();
            }
            else
            {
                throw new Exception("Unsupported AccelerationType. Supported types are CPU and GPU.");
            }
        }

        public void Train(double[][] inputs, double[][] expected, double learningRate, int epochs, bool silent = false)
        {
            // Build a *fixed* min/max from the entire training set:
            CalibrateInputs(inputs);

            long total = (long)inputs.Length * epochs;
            long count = 0;

            for (int e = 0; e < epochs; e++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    var xScaled = ScaleInput(inputs[i]);        // apply the same mapping every pass
                    Learn(xScaled, expected[i], learningRate);
                    count++;
                    if (!silent)
                        Terminal.Gui.Helpers.ProgressBarHelper.DisplayProgressBar(count, total, "Progress");
                }
            }
        }
        private double[] ScaleInput(double[] raw)
        {
            int n = raw.Length;
            var scaled = new double[n];
            var min = data.parameters.inputActivationMin;
            var max = data.parameters.inputActivationMax;

            for (int i = 0; i < n; i++)
            {
                double range = max[i] - min[i];
                scaled[i] = (range <= 0)
                    ? 0
                    : (raw[i] - min[i]) / range;
            }
            return scaled;
        }
        private void CalibrateInputs(double[][] inputs)
        {
            int inputSize = inputs[0].Length;
            var min = new double[inputSize];
            var max = new double[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                min[i] = double.PositiveInfinity;
                max[i] = double.NegativeInfinity;
            }

            foreach (var row in inputs)
            {
                for (int i = 0; i < inputSize; i++)
                {
                    min[i] = Math.Min(min[i], row[i]);
                    max[i] = Math.Max(max[i], row[i]);
                }
            }

            data.parameters.inputActivationMin = min;
            data.parameters.inputActivationMax = max;
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
        private void BackPropagate(double[] costDerivs, double learningRate)
        {
            int L = data.layers.Length - 1;

            var deltasByLayer = ComputeDeltas(costDerivs, L);
            var globalNorm = ComputeGlobalNorm(deltasByLayer);
            var scale = ComputeClipScale(globalNorm, data.parameters.GradientClippingThreshold);
            if (scale != 1.0) ScaleAllDeltas(deltasByLayer, scale);

            ApplyUpdates(deltasByLayer, learningRate);
        }

        #region Global Gradient Scale Clipping 

        private List<double[]> ComputeDeltas(double[] costDerivs, int outputLayerIdx)
        {
            var deltas = new List<double[]>();

            // output layer
            var outLayer = data.layers[outputLayerIdx];
            var outDeltas = accelerationManager.CalculateOutputGradients(costDerivs, outLayer.Derivatives);
            deltas.Add(outDeltas);

            // hidden layers, in reverse
            double[] next = outDeltas;
            for (int i = outputLayerIdx - 1; i > 0; i--)
            {
                var layerAbove = data.layers[i + 1];
                var layer = data.layers[i];
                var hidDeltas = accelerationManager.CalculateHiddenGradients(
                                       layerAbove.Weights,
                                       next,
                                       layer.Derivatives);
                deltas.Add(hidDeltas);
                next = hidDeltas;
            }

            // now deltas[0] == output, deltas[1] == last hidden, …
            return deltas;
        }

        private double ComputeGlobalNorm(List<double[]> deltas)
        {
            double sumSq = deltas
                .SelectMany(arr => arr)
                .Sum(v => v * v);
            return Math.Sqrt(sumSq);
        }

        private double ComputeClipScale(double globalNorm, double clipThreshold)
            => (globalNorm > clipThreshold)
                 ? clipThreshold / globalNorm
                 : 1.0;

        private void ScaleAllDeltas(List<double[]> deltas, double scale)
        {
            foreach (var arr in deltas)
                for (int i = 0; i < arr.Length; i++)
                    arr[i] *= scale;
        }
        private void ApplyUpdates(List<double[]> deltasByLayer, double learningRate)
        {
            int L = data.layers.Length - 1;
            double λ = data.parameters.L2RegulationLamda;

            // output layer
            {
                var layer = data.layers[L];
                var outDeltas = deltasByLayer[0];
                layer.Weights = accelerationManager.UpdateWeights(
                    layer.Weights,
                    outDeltas,
                    data.layers[L - 1].Activations,
                    learningRate,
                    λ                 // << passes λ into every update
                );
                layer.Biases = accelerationManager.UpdateBias(layer.Biases, outDeltas, learningRate);
            }

            // hidden layers
            for (int idx = 1; idx < deltasByLayer.Count; idx++)
            {
                int layerIdx = L - idx;
                var layer = data.layers[layerIdx];
                var deltas = deltasByLayer[idx];
                var prevActs = data.layers[layerIdx - 1].Activations;

                layer.Weights = accelerationManager.UpdateWeights(
                    layer.Weights,
                    deltas,
                    prevActs,
                    learningRate,
                    λ                 // << same λ again
                );
                layer.Biases = accelerationManager.UpdateBias(layer.Biases, deltas, learningRate);
            }
        }

        #endregion

        public void SetSensoryNeuronsValues(double[] inputValues)
        {
            var inputLayer = data.layers[0].Activations;
            Parallel.For(0, inputValues.Length, i =>
                inputLayer[i] = inputValues[i]
            );
        }
        public double[] Predict(double[] rawInput)
        {
            var xScaled = ScaleInput(rawInput);
            SetInputLayer(xScaled);
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

        public static NeuralNetwork Load(string FileName, AccelerationType accelerationType, int DeviceId= 0)
        {
            Data data = Data.Load(FileName);
            NeuralNetwork nn = new NeuralNetwork(data);
            data.parameters.AccelerationType = accelerationType;
            data.parameters.AccelerationDeviceId = DeviceId;
            nn.data = data;
            return nn;
        }
        public void Save(string FileName)
        {
            data.Save(FileName);
        }
    }
}
