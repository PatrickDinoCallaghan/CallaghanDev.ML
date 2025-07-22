using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.Utilities.Utilities.Diagnostics;
using Terminal.Gui.Helpers;

namespace CallaghanDev.ML
{
    public class NeuralNetwork
    {
        private Data data;
        private IAccelerationManager accelerationManager;
        private CostFunction CostFunctionDeriv { get; set; }
        private delegate float CostFunction(float value, float prediction);
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
                Console.WriteLine($"Using {data.parameters.AccelerationType} acceleration on device {data.parameters.AccelerationDeviceId}.");
                accelerationManager = new AccelerationGPU(data.parameters.AccelerationType, data.parameters.AccelerationDeviceId);
            }
            else if (data.parameters.AccelerationType == AccelerationType.CPU)
            {
                Console.WriteLine($"Using {data.parameters.AccelerationType} acceleration.");
                accelerationManager = new AccelerationCPU();
            }
            else if (data.parameters.AccelerationType == AccelerationType.MultiThreadCPU)
            {
                Console.WriteLine($"Using {data.parameters.AccelerationType} acceleration.");
                accelerationManager = new AccelerationMutliThreadCPU();
            }
            else
            {
                throw new Exception("Unsupported AccelerationType. Supported types are CPU and GPU.");
            }
        }

        public void Train(float[][] inputs, float[][] expected, float learningRate, int epochs, bool silent = false)
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
                    {
                        Terminal.Gui.Helpers.ProgressBarHelper.DisplayProgressBar(count, total, "Progress");
                    }
                }
            }
        }
        private float[] ScaleInput(float[] raw)
        {
            int n = raw.Length;
            var scaled = new float[n];
            var min = data.parameters.inputActivationMin;
            var max = data.parameters.inputActivationMax;

            for (int i = 0; i < n; i++)
            {
                float range = max[i] - min[i];
                scaled[i] = (range <= 0) ? 0 : (raw[i] - min[i]) / range;
            }
            return scaled;
        }
        private void CalibrateInputs(float[][] inputs)
        {
            int inputSize = inputs[0].Length;
            var min = new float[inputSize];
            var max = new float[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                min[i] = float.PositiveInfinity;
                max[i] = float.NegativeInfinity;
            }

            foreach (var row in inputs)
            {
                for (int i = 0; i < inputSize; i++)
                {
                    min[i] = MathF.Min(min[i], row[i]);
                    max[i] = MathF.Max(max[i], row[i]);
                }
            }

            data.parameters.inputActivationMin = min;
            data.parameters.inputActivationMax = max;
        }

        private void Learn(float[] x, float[] y, float LearningRate)
        {
            SetInputLayer(x);
            ForwardPropagate();

            var last = data.layers[^1];
            float[] costDerivs = new float[last.Size];
            for (int i = 0; i < last.Size; i++)
            {
                costDerivs[i] = CostFunctionDeriv(last.Activations[i], y[i]);
            }
            BackPropagate(costDerivs, LearningRate);
        }

        private void SetInputLayer(float[] x)
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
        private void BackPropagate(float[] costDerivs, float learningRate)
        {
            int L = data.layers.Length - 1;

            var deltasByLayer = ComputeDeltas(costDerivs, L);
            var globalNorm = ComputeGlobalNorm(deltasByLayer);
            var scale = ComputeClipScale(globalNorm, data.parameters.GradientClippingThreshold);
            if (scale != 1.0) ScaleAllDeltas(deltasByLayer, scale);

            ApplyUpdates(deltasByLayer, learningRate);
        }


        #region Global Gradient Scale Clipping 

        private List<float[]> ComputeDeltas(float[] costDerivs, int outputLayerIdx)
        {
            var deltas = new List<float[]>();

            // output layer
            var outLayer = data.layers[outputLayerIdx];
            var outDeltas = accelerationManager.CalculateOutputGradients(costDerivs, outLayer.Derivatives);
            deltas.Add(outDeltas);

            // hidden layers, in reverse
            float[] next = outDeltas;
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

        private float ComputeGlobalNorm(List<float[]> deltas)
        {
            float sumSq = deltas
                .SelectMany(arr => arr)
                .Sum(v => v * v);
            return MathF.Sqrt(sumSq);
        }

        private float ComputeClipScale(float globalNorm, float clipThreshold)
            => (globalNorm > clipThreshold)
                 ? clipThreshold / globalNorm
                 : 1.0f;

        private void ScaleAllDeltas(List<float[]> deltas, float scale)
        {
            foreach (var arr in deltas)
            {
                for (int i = 0; i < arr.Length; i++)
                {
                    arr[i] *= scale;
                }
            }
        }
        private void ApplyUpdates(List<float[]> deltasByLayer, float learningRate)
        {
            int L = data.layers.Length - 1;
            float λ = data.parameters.L2RegulationLamda;

            // output layer
            {
                var layer = data.layers[L];
                var outDeltas = deltasByLayer[0];
                layer.Weights = accelerationManager.UpdateWeights(
                    layer.Weights,
                    outDeltas,
                    data.layers[L - 1].Activations,
                    learningRate,
                    λ                 // << passes lambda into every update
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
                    λ                 // << same lambda again
                );
                layer.Biases = accelerationManager.UpdateBias(layer.Biases, deltas, learningRate);
            }
        }

        #endregion

        public void SetSensoryNeuronsValues(float[] inputValues)
        {
            var inputLayer = data.layers[0].Activations;
            Parallel.For(0, inputValues.Length, i =>
                inputLayer[i] = inputValues[i]
            );
        }
        public float[] Predict(float[] rawInput)
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
                    costFunction = (float ActualValue, float predicted) => { return MathF.Pow(predicted - ActualValue, 2); };
                    CostFunctionDeriv = (float ActualValue, float predicted) => { return 2 * (predicted - ActualValue); };
                    break;
                case CostFunctionType.ZeroWeightedMSE:
                    costFunction = (float ActualValue, float predicted) =>
                    {
                        float zeroWeight = 0.25f;
                        float nonZeroWeight = 1.0f;

                        // Determine the weight based on the true value
                        float weight = (ActualValue == 0) ? zeroWeight : nonZeroWeight;

                        // Calculate the weighted MSE for the single prediction
                        return weight * MathF.Pow(predicted - ActualValue, 2f);
                    };
                    CostFunctionDeriv = (float ActualValue, float predicted) =>
                    {
                        float zeroWeight = 0.25f;
                        float nonZeroWeight = 1.0f;

                        // Determine the weight based on the true value
                        float weight = (ActualValue == 0) ? zeroWeight : nonZeroWeight;

                        // Calculate the derivative of the weighted MSE for the single prediction
                        return 2 * weight * (predicted - ActualValue);
                    };
                    break;
                case CostFunctionType.binaryCrossEntropy:
                    costFunction = (float ActualValue, float predicted) => { return -(ActualValue * MathF.Log(predicted) + (1 - ActualValue) * MathF.Log(1 - predicted)); };
                    CostFunctionDeriv = (float ActualValue, float predicted) => { return (predicted - ActualValue) / (predicted * (1 - predicted)); };
                    break;
                case CostFunctionType.mae:
                    costFunction = (float ActualValue, float predicted) => { return MathF.Abs(ActualValue - predicted); };
                    CostFunctionDeriv = (float ActualValue, float predicted) => { return ActualValue > predicted ? -1 : 1; };
                    break;
                case CostFunctionType.huberLoss:
                    costFunction = (ActualValue, predicted) =>
                    {
                        float diff = ActualValue - predicted;
                        if (MathF.Abs(diff) <= data.parameters.HuberLossDelta)
                        {
                            return 0.5f * diff * diff;
                        }
                        else
                        {
                            return data.parameters.HuberLossDelta * (MathF.Abs(diff) - 0.5f * data.parameters.HuberLossDelta);
                        }
                    };
                    CostFunctionDeriv = (ActualValue, predicted) =>
                    {
                        float diff = ActualValue - predicted;
                        if (MathF.Abs(diff) <= data.parameters.HuberLossDelta)
                        {
                            return diff;
                        }
                        else
                        {
                            return data.parameters.HuberLossDelta * MathF.Sign(diff);
                        }
                    };
                    break;
                case CostFunctionType.categoricalCrossEntropy:
                    costFunction = (ActualValue, predicted) =>
                    {
                        return -ActualValue * MathF.Log(predicted + 1e-15f);
                    };
                    CostFunctionDeriv = (float ActualValue, float predicted) => { return predicted - ActualValue; };
                    break;
                default:
                    throw new ArgumentException("Unsupported cost function type");
            }
        }
        public float[] CalculateCost(float[] expectedOutputValues)
        {
            int columnIndex = data.layers.Length - 1;

            float[] costDifferences = new float[data.layers[columnIndex].Size];

            Parallel.For(0, data.layers[columnIndex].Size, i =>
            {
                float nnOutputValue = data.layers[columnIndex].Activations[i];
                costDifferences[i] = CostFunctionDeriv(nnOutputValue, expectedOutputValues[i]);
            });

            return costDifferences;
        }

        public static NeuralNetwork Load(string FileName, AccelerationType accelerationType, int DeviceId = 0)
        {
            Data data = Data.Load(FileName);
            data.parameters.AccelerationType = accelerationType;
            data.parameters.AccelerationDeviceId = DeviceId;
            NeuralNetwork nn = new NeuralNetwork(data);
            nn.ResetGpuBatchManager();
            return nn;
        }
        public void Save(string FileName)
        {
            data.Save(FileName);
        }

        #region Batch-Processing Methods
        private AccelerationGPUBatch GPUBatchAccelerationManager;

        public void TrainBatch(float[][] inputs, float[][] expected, int batchSize, float learningRate, int epochs, bool silent = false)
        {
            if (data.parameters.AccelerationType != AccelerationType.GPU && data.parameters.AccelerationType != AccelerationType.CUDA)
            {
                throw new InvalidOperationException("Batch training requires GPU/CUDA.");
            }

            if (GPUBatchAccelerationManager == null)
            {
                GPUBatchAccelerationManager = new AccelerationGPUBatch(accelerationManager, data.parameters.AccelerationType, data.parameters.AccelerationDeviceId);
            }

            if (batchSize <= 1)
            {
                Train(inputs, expected, learningRate, epochs, silent);
                return;
            }

            CalibrateInputs(inputs);

            int n = inputs.Length;
            long totalBatches = (long)MathF.Ceiling(n / (float)batchSize) * epochs;
            long count = 0;
            var rand = new Random();

            for (int e = 0; e < epochs; e++)
            {
                var idx = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).ToArray();

                for (int start = 0; start < n; start += batchSize)
                {
                    int size = Math.Min(batchSize, n - start);
                    var batchX = new float[size][];
                    var batchY = new float[size][];

                    for (int j = 0; j < size; j++)
                    {
                        batchX[j] = ScaleInput(inputs[idx[start + j]]);
                        batchY[j] = expected[idx[start + j]];
                    }

                    LearnBatch(batchX, batchY, learningRate);
                    count++;
                    if (!silent)
                    {
                        ProgressBarHelper.DisplayProgressBar(count, totalBatches, "Training Batches");
                    }
                }
            }

            for (int l = 1; l < data.layers.Length; l++)
            {
                var layer = data.layers[l];

                GPUBatchAccelerationManager.CopyLayerToCPU(l, layer.Weights, layer.Biases);
            }
        }

        /// <summary>
        /// Initializes the first-layer activations for a mini‑batch.
        /// </summary>
        /// <param name="xBatch">A [batchSize][inputSize] jagged array of scaled inputs.</param>
        private void SetInputBatch(float[][] xBatch)
        {
            if (xBatch == null || xBatch.Length == 0)
            {
                throw new ArgumentException("xBatch must be non‑empty.", nameof(xBatch));
            }
            data.layers[0].ActivationsBatch = xBatch;

            data.layers[0].DerivativesBatch = new float[xBatch.Length][];
            Parallel.For(0, xBatch.Length, i =>
            {
                data.layers[0].DerivativesBatch[i] = new float[data.layers[0].Size];
            });
        }

        private void LearnBatch(float[][] xBatch, float[][] yBatch, float lr)
        {
            SetInputBatch(xBatch);

            ForwardPropagateBatch();

            var outL = data.layers[^1];
            int B = yBatch.Length;
            var costDeriv = new float[B][];

            Parallel.For(0, B, i =>
            {
                var deriv = new float[outL.Size];
                for (int j = 0; j < outL.Size; j++)
                {
                    deriv[j] = CostFunctionDeriv(outL.ActivationsBatch[i][j], yBatch[i][j]);
                }
                costDeriv[i] = deriv;
            });

            BackPropagateBatch(costDeriv, lr);
        }

        private void ForwardPropagateBatch()
        {
            for (int l = 1; l < data.layers.Length; l++)
            {
                var prev = data.layers[l - 1];
                var cur = data.layers[l];

                var (acts, ders) = GPUBatchAccelerationManager.CalculateBatch(
                    prev.ActivationsBatch,
                    cur.Weights,
                    cur.Biases,
                    cur.ActivationType,
                    l
                );

                cur.ActivationsBatch = acts;
                cur.DerivativesBatch = ders;
            }
        }

        private void BackPropagateBatch(float[][] costDerivBatch, float lr)
        {
            int L = data.layers.Length - 1;
            var deltas = ComputeBatchDeltas(costDerivBatch, L);
            var norm = ComputeBatchGlobalNorm(deltas);
            var scale = ComputeClipScale(norm, data.parameters.GradientClippingThreshold);
            if (scale != 1.0)
            {
                ScaleAllBatchDeltas(deltas, scale);
            }
            ApplyBatchUpdates(deltas, lr);
        }

        private List<float[][]> ComputeBatchDeltas(float[][] costDerivBatch, int outIdx)
        {
            var list = new List<float[][]>();
            var outLayer = data.layers[outIdx];
            list.Add(GPUBatchAccelerationManager.CalculateOutputGradients(costDerivBatch, outLayer.DerivativesBatch));
            var next = list[0];
            for (int i = outIdx - 1; i > 0; i--)
            {
                var layer = data.layers[i];
                var above = data.layers[i + 1];
                list.Add(GPUBatchAccelerationManager.CalculateHiddenGradients(above.Weights, next, layer.DerivativesBatch));
                next = list[^1];
            }
            return list;
        }

        private float ComputeBatchGlobalNorm(List<float[][]> deltas)
        {
            float sumSq = deltas
           .AsParallel()
           .SelectMany(mat => mat)
           .SelectMany(row => row)
           .Select(val => val * val)
           .Sum();
            return MathF.Sqrt(sumSq);
        }
        private void ScaleAllBatchDeltas(List<float[][]> deltas, float scale)
        {
            Parallel.For(0, deltas.Count, idx =>
            {
                var mat = deltas[idx];
                for (int i = 0; i < mat.Length; i++)
                {
                    for (int j = 0; j < mat[i].Length; j++)
                    {
                        mat[i][j] *= scale;
                    }
                }

            });
        }
        private void ApplyBatchUpdates(List<float[][]> deltas, float lr)
        {
            int L = data.layers.Length - 1;
            float λ = data.parameters.L2RegulationLamda;

            // deltas[0] is for the output layer, deltas[1] for the last hidden, etc.
            for (int idx = 0; idx < deltas.Count; idx++)
            {
                int layerIdx = L - idx;
                var layer = data.layers[layerIdx];

                // prevActsBatch is null only for the very first (input) layer
                float[][] prevActs = layerIdx > 0 ? data.layers[layerIdx - 1].ActivationsBatch : null;

                GPUBatchAccelerationManager.ApplyBatchUpdatesGPU(
                    layer.Weights,      // CPU array for weights (for backing-up / copy-back)
                    deltas[idx],        // the [batchSize][neurons] delta array
                    prevActs,           // the [batchSize][prevNeurons] activations
                    lr,
                    λ,
                    layerIdx
                );


            }
        }

        #endregion
        public void ResetGpuBatchManager()
        {
            GPUBatchAccelerationManager?.Dispose();
            GPUBatchAccelerationManager = null;
        }

    }
}