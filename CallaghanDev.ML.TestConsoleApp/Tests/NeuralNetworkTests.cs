using CallaghanDev.ML;
using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    internal sealed class NeuralNetworkTests : TestBase
    {
        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * Neural Network");
        }

        private (Action test, string name)[] Tests() => new (Action test, string name)[]
        {
            (Test_NeuralNetworkXorTest, "NeuralNetwork: XOR learns expected labels"),
            (Test_NeuralNetworkTwoSpiralsTest, "NeuralNetwork: two spirals train/predict smoke test"),
            (Test_NeuralNetworkAndGPUTest, "NeuralNetwork: AND original GPU-labelled config learns labels"),
            (Test_NeuralNetworkAndCPUTest, "NeuralNetwork: AND CPU learns expected labels"),
            (Test_NeuralNetworkOrTest, "NeuralNetwork: OR learns expected labels"),
            (Test_MnistCsvSmokeTest_WhenFileExists, "NeuralNetwork: MNIST csv smoke test when available"),
        };

        private void Test_NeuralNetworkXorTest()
        {
            float[][] inputs = BinaryInputs();
            float[][] expectedOutputs = new float[][]
            {
                new float[] { 0f },
                new float[] { 1f },
                new float[] { 1f },
                new float[] { 0f }
            };

            AssertLearnsTruthTable(
                testName: "XOR",
                networkFactory: () => NewBinaryNetwork(
                    accelerationType: AccelerationType.CPU,
                    activationType: ActivationType.Leakyrelu,
                    hiddenLayers: 1,
                    hiddenLayerWidth: 4,
                    gradientClippingThreshold: 1f,
                    l2RegulationLamda: 0.01f),
                inputs: inputs,
                expectedOutputs: expectedOutputs,
                learningRate: 0.1f,
                epochs: 1000,
                threshold: 0.1f,
                attempts: 5);
        }

        private void Test_NeuralNetworkTwoSpiralsTest()
        {
            const int pointsPerSpiral = 500;
            const int sampleChecks = 10;

            CreateTwoSpirals(
                pointsPerSpiral,
                out float[][] inputs,
                out float[][] expectedOutputs,
                out float maxTheta);

            var neuralNetwork = NewNetwork(
                inputCount: 2,
                inputMin: -maxTheta,
                inputMax: maxTheta,
                accelerationType: AccelerationType.CPU,
                hiddenLayers: 2,
                hiddenLayerWidth: 16,
                outputCount: 1,
                activationType: ActivationType.Leakyrelu,
                costFunction: CostFunctionType.mse,
                gradientClippingThreshold: 1f,
                l2RegulationLamda: 0.01f);

            float beforeLoss = MeanSquaredError(neuralNetwork, inputs, expectedOutputs, "TwoSpirals before training");
            TrainNetwork(neuralNetwork, inputs, expectedOutputs, 0.01f, 5000);
            float afterLoss = MeanSquaredError(neuralNetwork, inputs, expectedOutputs, "TwoSpirals after training");

            AssertFinite(beforeLoss, "TwoSpirals loss before training");
            AssertFinite(afterLoss, "TwoSpirals loss after training");

            // The original two-spirals test only printed sample predictions. Keep this as a
            // smoke test: training must run and predictions must stay finite.
            var rng = new Random(42);
            for (int i = 0; i < sampleChecks; i++)
            {
                int sampleIndex = rng.Next(inputs.Length);
                float[] prediction = PredictNetwork(neuralNetwork, inputs[sampleIndex]);

                Assert(prediction != null, $"TwoSpirals prediction was null at sample {sampleIndex}");
                Assert(prediction.Length == 1,
                    $"TwoSpirals prediction length mismatch at sample {sampleIndex}: {prediction.Length}");
                AssertFinite(prediction[0], $"TwoSpirals prediction[{sampleIndex}]");
            }
        }

        private void Test_NeuralNetworkAndGPUTest()
        {
            float[][] inputs = BinaryInputs();
            float[][] expectedOutputs = new float[][]
            {
                new float[] { 0f },
                new float[] { 0f },
                new float[] { 0f },
                new float[] { 1f }
            };

            // The original method was named GPU, but its Parameters used AccelerationType.CPU.
            // This preserves the original executable setup while making it assert pass/fail.
            AssertLearnsTruthTable(
                testName: "AND GPU-labelled original config",
                networkFactory: () => NewBinaryNetwork(
                    accelerationType: AccelerationType.CPU,
                    activationType: ActivationType.Tanh,
                    hiddenLayers: 1,
                    hiddenLayerWidth: 4,
                    gradientClippingThreshold: 1f),
                inputs: inputs,
                expectedOutputs: expectedOutputs,
                learningRate: 0.01f,
                epochs: 1000,
                threshold: 0.5f,
                attempts: 5);
        }

        private void Test_NeuralNetworkAndCPUTest()
        {
            float[][] inputs = BinaryInputs();
            float[][] expectedOutputs = new float[][]
            {
                new float[] { 0f },
                new float[] { 0f },
                new float[] { 0f },
                new float[] { 1f }
            };

            AssertLearnsTruthTable(
                testName: "AND CPU",
                networkFactory: () => NewBinaryNetwork(
                    accelerationType: AccelerationType.CPU,
                    activationType: ActivationType.Tanh,
                    hiddenLayers: 1,
                    hiddenLayerWidth: 4,
                    gradientClippingThreshold: 1f),
                inputs: inputs,
                expectedOutputs: expectedOutputs,
                learningRate: 0.01f,
                epochs: 1000,
                threshold: 0.5f,
                attempts: 5);
        }

        private void Test_NeuralNetworkOrTest()
        {
            float[][] inputs = BinaryInputs();
            float[][] expectedOutputs = new float[][]
            {
                new float[] { 0f },
                new float[] { 1f },
                new float[] { 1f },
                new float[] { 1f }
            };

            AssertLearnsTruthTable(
                testName: "OR",
                networkFactory: () => NewBinaryNetwork(
                    accelerationType: AccelerationType.CPU,
                    activationType: ActivationType.Tanh,
                    hiddenLayers: 1,
                    hiddenLayerWidth: 4,
                    gradientClippingThreshold: 1f,
                    activationDistribution: ActivationDistribution.Uniform),
                inputs: inputs,
                expectedOutputs: expectedOutputs,
                learningRate: 0.01f,
                epochs: 1000,
                threshold: 0.5f,
                attempts: 5);
        }

        private void Test_MnistCsvSmokeTest_WhenFileExists()
        {
            string mnistPath = FindOptionalFile("mnist_train.csv");

            if (mnistPath == null)
            {
                Console.WriteLine("         ↳ mnist_train.csv not found; optional MNIST smoke test skipped.");
                return;
            }

            LoadMnistCsv(
                path: mnistPath,
                maxRows: 1000,
                out float[][] trainingInputs,
                out float[][] trainingOutputs);

            Assert(trainingInputs.Length > 0, "MNIST csv loaded no rows");
            Assert(trainingInputs.Length == trainingOutputs.Length, "MNIST input/output row count mismatch");
            Assert(trainingInputs.All(row => row.Length == 784), "MNIST input rows must contain 784 pixels");
            Assert(trainingOutputs.All(row => row.Length == 10), "MNIST output rows must contain 10 classes");

            NeuralNetwork neuralNetwork;

            try
            {
                neuralNetwork = NewNetwork(
                    inputCount: 784,
                    inputMin: 0f,
                    inputMax: 1f,
                    accelerationType: AccelerationType.GPU,
                    hiddenLayers: 10,
                    hiddenLayerWidth: 784,
                    outputCount: 10,
                    activationType: ActivationType.Leakyrelu,
                    costFunction: CostFunctionType.mse);

                TrainNetwork(neuralNetwork, trainingInputs, trainingOutputs, 0.01f, 15);
            }
            catch (Exception ex) when (LooksLikeUnavailableAcceleration(ex))
            {
                Console.WriteLine($"         ↳ GPU acceleration unavailable ({Unwrap(ex).GetType().Name}); retrying MNIST smoke test on CPU.");

                neuralNetwork = NewNetwork(
                    inputCount: 784,
                    inputMin: 0f,
                    inputMax: 1f,
                    accelerationType: AccelerationType.CPU,
                    hiddenLayers: 10,
                    hiddenLayerWidth: 784,
                    outputCount: 10,
                    activationType: ActivationType.Leakyrelu,
                    costFunction: CostFunctionType.mse);

                TrainNetwork(neuralNetwork, trainingInputs, trainingOutputs, 0.01f, 15);
            }

            float[] prediction = PredictNetwork(neuralNetwork, trainingInputs[0]);

            Assert(prediction != null, "MNIST prediction was null");
            Assert(prediction.Length == 10,
                $"MNIST prediction should have 10 outputs, got {prediction.Length}");

            for (int i = 0; i < prediction.Length; i++)
            {
                AssertFinite(prediction[i], $"MNIST prediction[{i}]");
            }

            int predictedLabel = IndexOfMax(prediction);
            int expectedLabel = IndexOfMax(trainingOutputs[0]);

            Assert(predictedLabel >= 0 && predictedLabel <= 9,
                $"MNIST predicted label out of range: {predictedLabel}; expected label was {expectedLabel}");
        }

        private void AssertLearnsTruthTable(
            string testName,
            Func<NeuralNetwork> networkFactory,
            float[][] inputs,
            float[][] expectedOutputs,
            float learningRate,
            int epochs,
            float threshold,
            int attempts)
        {
            var failures = new List<string>();

            for (int attempt = 1; attempt <= attempts; attempt++)
            {
                try
                {
                    NeuralNetwork neuralNetwork = networkFactory();

                    float beforeLoss = MeanSquaredError(neuralNetwork, inputs, expectedOutputs, $"{testName} before training");
                    TrainNetwork(neuralNetwork, inputs, expectedOutputs, learningRate, epochs);
                    float afterLoss = MeanSquaredError(neuralNetwork, inputs, expectedOutputs, $"{testName} after training");

                    AssertFinite(beforeLoss, $"{testName} loss before training");
                    AssertFinite(afterLoss, $"{testName} loss after training");

                    float accuracy = TruthTableAccuracy(neuralNetwork, inputs, expectedOutputs, threshold);
                    Assert(afterLoss < beforeLoss || accuracy >= 1f,
                        $"{testName} did not improve loss enough. loss {beforeLoss:F6} → {afterLoss:F6}; accuracy={accuracy:P0}");

                    AssertBinaryTruthTable(neuralNetwork, inputs, expectedOutputs, threshold, testName);
                    return;
                }
                catch (Exception ex)
                {
                    failures.Add($"attempt {attempt}: {Unwrap(ex).Message}");
                }
            }

            Assert(false, $"{testName} failed after {attempts} attempt(s). {string.Join(" | ", failures)}");
        }

        private void AssertBinaryTruthTable(
            NeuralNetwork neuralNetwork,
            float[][] inputs,
            float[][] expectedOutputs,
            float threshold,
            string testName)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                float[] prediction = PredictNetwork(neuralNetwork, inputs[i]);

                Assert(prediction != null, $"{testName} prediction was null at row {i}");
                Assert(prediction.Length == expectedOutputs[i].Length,
                    $"{testName} prediction length mismatch at row {i}: expected {expectedOutputs[i].Length}, got {prediction.Length}");

                for (int j = 0; j < prediction.Length; j++)
                {
                    AssertFinite(prediction[j], $"{testName} prediction[{i}][{j}]");
                }

                int predictedLabel = prediction[0] >= threshold ? 1 : 0;
                int expectedLabel = expectedOutputs[i][0] >= 0.5f ? 1 : 0;

                Assert(predictedLabel == expectedLabel,
                    $"{testName} row {i} [{string.Join(", ", inputs[i])}] expected {expectedLabel}, " +
                    $"predicted {predictedLabel}, raw={prediction[0]:F6}, threshold={threshold:F3}");
            }
        }

        private float MeanSquaredError(
            NeuralNetwork neuralNetwork,
            float[][] inputs,
            float[][] expectedOutputs,
            string name)
        {
            Assert(inputs != null, $"{name}: inputs were null");
            Assert(expectedOutputs != null, $"{name}: expected outputs were null");
            Assert(inputs.Length == expectedOutputs.Length,
                $"{name}: input/output row count mismatch: {inputs.Length} != {expectedOutputs.Length}");

            double sum = 0d;
            int count = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                float[] prediction = PredictNetwork(neuralNetwork, inputs[i]);

                Assert(prediction != null, $"{name}: prediction was null at row {i}");
                Assert(prediction.Length == expectedOutputs[i].Length,
                    $"{name}: prediction length mismatch at row {i}: expected {expectedOutputs[i].Length}, got {prediction.Length}");

                for (int j = 0; j < prediction.Length; j++)
                {
                    AssertFinite(prediction[j], $"{name} prediction[{i}][{j}]");

                    double diff = prediction[j] - expectedOutputs[i][j];
                    sum += diff * diff;
                    count++;
                }
            }

            Assert(count > 0, $"{name}: no outputs to score");
            return (float)(sum / count);
        }

        private float TruthTableAccuracy(
            NeuralNetwork neuralNetwork,
            float[][] inputs,
            float[][] expectedOutputs,
            float threshold)
        {
            int correct = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                float[] prediction = PredictNetwork(neuralNetwork, inputs[i]);
                Assert(prediction != null && prediction.Length > 0,
                    $"Truth table prediction missing at row {i}");

                int predictedLabel = prediction[0] >= threshold ? 1 : 0;
                int expectedLabel = expectedOutputs[i][0] >= 0.5f ? 1 : 0;

                if (predictedLabel == expectedLabel)
                {
                    correct++;
                }
            }

            return inputs.Length == 0 ? 0f : (float)correct / inputs.Length;
        }

        private static NeuralNetwork NewBinaryNetwork(
            AccelerationType accelerationType,
            ActivationType activationType,
            int hiddenLayers,
            int hiddenLayerWidth,
            float gradientClippingThreshold,
            float l2RegulationLamda = 0f,
            ActivationDistribution? activationDistribution = null)
        {
            return NewNetwork(
                inputCount: 2,
                inputMin: 0f,
                inputMax: 1f,
                accelerationType: accelerationType,
                hiddenLayers: hiddenLayers,
                hiddenLayerWidth: hiddenLayerWidth,
                outputCount: 1,
                activationType: activationType,
                costFunction: CostFunctionType.mse,
                gradientClippingThreshold: gradientClippingThreshold,
                l2RegulationLamda: l2RegulationLamda,
                activationDistribution: activationDistribution);
        }

        private static NeuralNetwork NewNetwork(
            int inputCount,
            float inputMin,
            float inputMax,
            AccelerationType accelerationType,
            int hiddenLayers,
            int hiddenLayerWidth,
            int outputCount,
            ActivationType activationType,
            CostFunctionType costFunction,
            float gradientClippingThreshold = 1f,
            float l2RegulationLamda = 0f,
            ActivationDistribution? activationDistribution = null,
            ActivationType? outputActivationType = null)
        {
            Parameters parameters = NewParameters(
                inputCount: inputCount,
                inputMin: inputMin,
                inputMax: inputMax,
                accelerationType: accelerationType,
                hiddenLayers: hiddenLayers,
                hiddenLayerWidth: hiddenLayerWidth,
                outputCount: outputCount,
                activationType: activationType,
                costFunction: costFunction,
                gradientClippingThreshold: gradientClippingThreshold,
                l2RegulationLamda: l2RegulationLamda,
                activationDistribution: activationDistribution,
                outputActivationType: outputActivationType);

            return new NeuralNetwork(parameters);
        }

        private static Parameters NewParameters(
            int inputCount,
            float inputMin,
            float inputMax,
            AccelerationType accelerationType,
            int hiddenLayers,
            int hiddenLayerWidth,
            int outputCount,
            ActivationType activationType,
            CostFunctionType costFunction,
            float gradientClippingThreshold = 1f,
            float l2RegulationLamda = 0f,
            ActivationDistribution? activationDistribution = null,
            ActivationType? outputActivationType = null)
        {
            if (inputCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(inputCount), "Input count must be positive.");
            }

            if (hiddenLayers < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(hiddenLayers), "Hidden layer count cannot be negative.");
            }

            if (hiddenLayers > 0 && hiddenLayerWidth <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(hiddenLayerWidth), "Hidden layer width must be positive when hidden layers are requested.");
            }

            if (outputCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputCount), "Output count must be positive.");
            }

            var layerWidths = new List<int> { inputCount };
            var layerActivations = new List<ActivationType> { ActivationType.None };

            for (int i = 0; i < hiddenLayers; i++)
            {
                layerWidths.Add(hiddenLayerWidth);
                layerActivations.Add(activationType);
            }

            layerWidths.Add(outputCount);
            layerActivations.Add(outputActivationType ?? activationType);

            var parameters = new Parameters
            {
                AccelerationType = accelerationType,
                AccelerationDeviceId = 0,
                CostFunction = costFunction,
                ActivationDistribution = activationDistribution ?? ActivationDistribution.Normal,
                L2RegulationLamda = l2RegulationLamda,
                GradientClippingThreshold = gradientClippingThreshold,
                LayerWidths = layerWidths,
                LayerActivations = layerActivations,
            };

            SetInputActivationRange(parameters, inputCount, inputMin, inputMax);

            return parameters;
        }

        private static void SetInputActivationRange(
            Parameters parameters,
            int inputCount,
            float inputMin,
            float inputMax)
        {
            float lo = Math.Min(inputMin, inputMax);
            float hi = Math.Max(inputMin, inputMax);

            float[] mins = Enumerable.Repeat(lo, inputCount).ToArray();
            float[] maxs = Enumerable.Repeat(hi, inputCount).ToArray();

            SetParametersProperty(parameters, "inputActivationMin", mins);
            SetParametersProperty(parameters, "inputActivationMax", maxs);
        }

        private static void SetParametersProperty<T>(Parameters parameters, string propertyName, T value)
        {
            PropertyInfo property = typeof(Parameters).GetProperty(
                propertyName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

            if (property != null && property.CanWrite)
            {
                property.SetValue(parameters, value);
            }
        }

        private static void TrainNetwork(
            NeuralNetwork neuralNetwork,
            IList<float[]> inputs,
            IList<float[]> expectedOutputs,
            float learningRate,
            int epochs)
        {
            neuralNetwork.Train(inputs, expectedOutputs, learningRate, epochs);
        }

        private static float[] PredictNetwork(NeuralNetwork neuralNetwork, float[] input)
        {
            return neuralNetwork.Predict(input).ToArray();
        }

        private static float[][] BinaryInputs() => new float[][]
        {
            new float[] { 0f, 0f },
            new float[] { 0f, 1f },
            new float[] { 1f, 0f },
            new float[] { 1f, 1f }
        };

        private static void CreateTwoSpirals(
            int pointsPerSpiral,
            out float[][] inputs,
            out float[][] expectedOutputs,
            out float maxTheta)
        {
            inputs = new float[pointsPerSpiral * 2][];
            expectedOutputs = new float[pointsPerSpiral * 2][];

            maxTheta = 5f * MathF.PI;
            float step = maxTheta / (pointsPerSpiral - 1);

            int index = 0;
            for (int i = 0; i < pointsPerSpiral; i++)
            {
                float t = i * step;

                float x1 = t * MathF.Cos(t);
                float y1 = t * MathF.Sin(t);
                inputs[index] = new float[] { x1, y1 };
                expectedOutputs[index] = new float[] { 0f };
                index++;

                float x2 = t * MathF.Cos(t + MathF.PI);
                float y2 = t * MathF.Sin(t + MathF.PI);
                inputs[index] = new float[] { x2, y2 };
                expectedOutputs[index] = new float[] { 1f };
                index++;
            }
        }

        private static void LoadMnistCsv(
            string path,
            int maxRows,
            out float[][] inputs,
            out float[][] expectedOutputs)
        {
            var inputRows = new List<float[]>();
            var outputRows = new List<float[]>();

            using (var reader = new StreamReader(path))
            {
                while (!reader.EndOfStream && inputRows.Count < maxRows)
                {
                    string line = reader.ReadLine();
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

                    string[] values = line.Split(',');
                    if (values.Length != 785)
                    {
                        throw new InvalidDataException($"MNIST row should have 785 columns, got {values.Length}");
                    }

                    int label = Convert.ToInt32(values[0], CultureInfo.InvariantCulture);
                    if (label < 0 || label > 9)
                    {
                        throw new InvalidDataException($"MNIST label out of range: {label}");
                    }

                    var output = new float[10];
                    output[label] = 1f;
                    outputRows.Add(output);

                    float[] input = values
                        .Skip(1)
                        .Select(str => Convert.ToSingle(str, CultureInfo.InvariantCulture) / 255f)
                        .ToArray();

                    inputRows.Add(input);
                }
            }

            inputs = inputRows.ToArray();
            expectedOutputs = outputRows.ToArray();
        }

        private static int IndexOfMax(float[] values)
        {
            if (values == null || values.Length == 0)
            {
                return -1;
            }

            int bestIndex = 0;
            float bestValue = values[0];

            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > bestValue)
                {
                    bestValue = values[i];
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        private static string FindOptionalFile(string fileName)
        {
            var candidates = new List<string>
            {
                Path.Combine(Directory.GetCurrentDirectory(), fileName),
                Path.Combine(AppContext.BaseDirectory, fileName)
            };

            DirectoryInfo directory = new DirectoryInfo(Directory.GetCurrentDirectory());
            for (int i = 0; i < 6 && directory != null; i++, directory = directory.Parent)
            {
                candidates.Add(Path.Combine(directory.FullName, fileName));
            }

            foreach (string candidate in candidates.Distinct())
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            return null;
        }

        private static bool LooksLikeUnavailableAcceleration(Exception ex)
        {
            Exception unwrapped = Unwrap(ex);
            string text = (unwrapped.Message ?? string.Empty).ToLowerInvariant();

            return unwrapped is NotSupportedException ||
                   unwrapped is NotImplementedException ||
                   text.Contains("gpu") ||
                   text.Contains("cuda") ||
                   text.Contains("device") ||
                   text.Contains("accelerat");
        }

        private static Exception Unwrap(Exception ex)
        {
            while (ex is TargetInvocationException targetInvocationException && targetInvocationException.InnerException != null)
            {
                ex = targetInvocationException.InnerException;
            }

            return ex;
        }
    }
}
