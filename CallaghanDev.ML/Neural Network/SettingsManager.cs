using CallaghanDev.ML.Neural_Network.Exceptions;
using CallaghanDev.Utilities.ConsoleHelper;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.NN
{
    public class SettingsManager
    {

        private double Evaluate(NeuralNetwork nn, double[][] inputs, double[][] outputs)
        {
            double totalError = 0.0;
            List<double[]> predictions = new List<double[]>();

            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = nn.Predict(inputs[i]);
                predictions.Add(prediction);
                totalError += CalculateError(prediction, outputs[i]);
            }
            double InputSensitive = InputSensitivity(inputs, predictions.ToArray());

            return (inputs.Length / totalError) * InputSensitivity(inputs, predictions.ToArray());
        }

        private double InputSensitivity(double[][] inputs, double[][] outputs)
        {
            double totalError = 0.0;

            List<double[]> CheckedArray = new List<double[]>();
            List<double> SensitivityList = new List<double>();

            for (int i = 1; i < inputs.Length; i++)
            {
                if (CheckedArray.Where(r => ArrayIsEqual(r, inputs[i]) == true).Count() == 0)
                {
                    double Dec = 0;
                    double SensitivityCurrent = 0;

                    for (int j = 0; j < outputs[i].Length; j++)
                    {
                        SensitivityCurrent += Math.Abs(outputs[i][j] - outputs[i - 1][j]);
                        Dec += outputs[i][j];
                    }

                    SensitivityList.Add(SensitivityCurrent / Dec);
                }
            }

            double Sensitivity = SensitivityList.Average();
            return Sensitivity;
        }

        private bool ArrayIsEqual(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length)
            {
                return false;
            }
            else
            {
                for (int i = 0; i < array1.Length; i++)
                {
                    if (array1[i] != array2[i])
                    {
                        return false;
                    }
                }
                return true;
            }
        }

        private double CalculateError(double[] prediction, double[] actual)
        {
            double error = 0.0;
            for (int i = 0; i < prediction.Length; i++)
            {
                error += Math.Pow(prediction[i] - actual[i], 2);
            }
            return error;
        }

        public Tuple<Dictionary<string, object>, NeuralNetwork> GetBestSettings(double TrainingRate, double[][] trainingInputs, double[][] trainingOutputs, double[][] validationInputs, double[][] validationOutputs, int epochs = 1000)
        {
            Console.WriteLine("Get Best Settings");
            var bestSettings = new ConcurrentDictionary<string, object>();
            double bestPerformance = double.MaxValue;
            object bestPerformanceLock = new object();

            var activationTypes = Enum.GetValues(typeof(ActivationType)).Cast<ActivationType>().ToList();
            var costFunctionTypes = Enum.GetValues(typeof(CostFunctionType)).Cast<CostFunctionType>().ToList();

            var huberLossDeltas = new float[] { 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f };
            var l2RegulationLambdas = new double[] { 0, 0.001, 0.01, 0.1, 1.0, 10.0 };
            var clippingLimits = new float[] { 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f };

            long counter = 0;
            long maxVal = costFunctionTypes.Count * costFunctionTypes.Count * huberLossDeltas.Length * l2RegulationLambdas.Length * clippingLimits.Length;

            Console.WriteLine($"Number of trainable networks {maxVal}");

            NeuralNetwork bestNeuralNetwork = null;
            object bestNetworkLock = new object();

            var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

            Parallel.For(0, activationTypes.Count(), i =>
            {
                foreach (var costFunction in costFunctionTypes)
                {
                    foreach (var l2Lambda in l2RegulationLambdas)
                    {
                        foreach (var upperLimit in clippingLimits)
                        {
                            foreach (var lowerLimit in clippingLimits)
                            {
                                try
                                {
                                    if (costFunction == CostFunctionType.huberLoss)
                                    {
                                        foreach (var huberDelta in huberLossDeltas)
                                        {
                                            Parameters nParameters = new Parameters()
                                            {
                                                AccelerationType = AccelerationType.GPU,
                                                SensoryNeurons = DataManager.GetSensoryNeurons(trainingInputs),
                                                NoHiddenLayers = 5,
                                                HiddenLayerWidth = trainingInputs[0].Length,
                                                NumberOfOutputs = trainingOutputs[0].Length,
                                                DefaultActivationType = activationTypes[(int)i],
                                                CostFunction = costFunction,
                                                HuberLossDelta = huberDelta,
                                                L2RegulationLamda = l2Lambda,
                                                ClippingLimitUpper = upperLimit,
                                                ClippingLimitLower = -lowerLimit,
                                            };

                                            NeuralNetwork nn = new NeuralNetwork(nParameters);

                                            nn.Train(trainingInputs, trainingOutputs, TrainingRate, epochs, true);
                                            var performance = Evaluate(nn, validationInputs, validationOutputs);

                                            lock (bestPerformanceLock)
                                            {
                                                if (performance > bestPerformance)
                                                {
                                                    bestPerformance = performance;
                                                    bestSettings["ActivationType"] = activationTypes[(int)i];
                                                    bestSettings["CostFunction"] = costFunction;
                                                    bestSettings["L2RegulationLambda"] = l2Lambda;
                                                    bestSettings["ClippingLimit_Upper"] = upperLimit;
                                                    bestSettings["ClippingLimit_Lower"] = lowerLimit;
                                                    bestSettings["HuberLossDelta"] = huberDelta;

                                                    Debug.WriteLine($"Best Performance: {bestPerformance}");

                                                    foreach (var setting in bestSettings)
                                                    {
                                                        Debug.WriteLine($"{setting.Key}: {setting.Value}");
                                                    }

                                                    lock (bestNetworkLock)
                                                    {
                                                        bestNeuralNetwork?.Dispose();
                                                        bestNeuralNetwork = nn;
                                                    }
                                                }
                                            }

                                            Interlocked.Increment(ref counter);
                                            cnsl.DisplayProgressBar(counter, maxVal, " Progress");
                                        }
                                    }

                                    else
                                    {
                                        Parameters nParameters = new Parameters()
                                        {
                                            AccelerationType = AccelerationType.GPU,
                                            SensoryNeurons = DataManager.GetSensoryNeurons(trainingInputs),
                                            NoHiddenLayers = 5,
                                            HiddenLayerWidth = trainingInputs[0].Length,
                                            NumberOfOutputs = trainingOutputs[0].Length,
                                            DefaultActivationType = activationTypes[(int)i],
                                            CostFunction = costFunction,
                                            L2RegulationLamda = l2Lambda,
                                            ClippingLimitUpper = upperLimit,
                                            ClippingLimitLower = -lowerLimit,
                                        };

                                        NeuralNetwork nn = new NeuralNetwork(nParameters);
                                        nn.Train(trainingInputs, trainingOutputs, TrainingRate, epochs, true);
                                        var performance = Evaluate(nn, validationInputs, validationOutputs);

                                        lock (bestPerformanceLock)
                                        {
                                            if (performance > bestPerformance)
                                            {
                                                bestPerformance = performance;
                                                bestSettings["ActivationType"] = activationTypes[(int)i];
                                                bestSettings["CostFunction"] = costFunction;
                                                bestSettings["L2RegulationLambda"] = l2Lambda;
                                                bestSettings["ClippingLimit_Upper"] = upperLimit;
                                                bestSettings["ClippingLimit_Lower"] = lowerLimit;

                                                Debug.WriteLine($"Best Performance: {bestPerformance}");
                                                foreach (var setting in bestSettings)
                                                {
                                                    Debug.WriteLine($"{setting.Key}: {setting.Value}");
                                                }

                                                lock (bestNetworkLock)
                                                {
                                                    bestNeuralNetwork?.Dispose();
                                                    bestNeuralNetwork = nn;
                                                }
                                            }
                                        }

                                        Interlocked.Increment(ref counter);
                                        cnsl.DisplayProgressBar(counter, maxVal, " Progress");

                                    }
                                }
                                catch (NaNException nanEx)
                                {
                                    Debug.WriteLine($"{nanEx.Message} \n activationType:{activationTypes[(int)i]}, costFunction:{costFunction.ToString()}, clippingLimits:[{lowerLimit},{upperLimit}], l2Lambda:{l2Lambda}");
                                    continue; // Skip the rest of the loop body and move to the next iteration
                                }
                                catch (InfinityException infinityEx)
                                {
                                    Debug.WriteLine($"{infinityEx.Message} \n activationType:{activationTypes[(int)i]}, costFunction:{costFunction.ToString()}, clippingLimits:[{lowerLimit},{upperLimit}], l2Lambda:{l2Lambda}");
                                    continue; // Skip the rest of the loop body and move to the next iteration
                                }
                                catch (Exception ex)
                                {
                                    Debug.WriteLine(ex.Message);
                                    continue; // Skip the rest of the loop body and move to the next iteration
                                }
                            }
                        }
                    }
                }
            });

            Console.WriteLine($"Best Performance: {bestPerformance}");
            foreach (var setting in bestSettings)
            {
                Console.WriteLine($"{setting.Key}: {setting.Value}");
            }

            return new Tuple<Dictionary<string, object>, NeuralNetwork>(new Dictionary<string, object>(bestSettings), bestNeuralNetwork);

        }

    }
}
