using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AutoTuning
{
    public class TrainingDiagnostics
    {
        public List<float> TrainingLosses { get; } = new();
        public List<float> ValidationLosses { get; } = new();
        public List<float[]> ValidationPredictions { get; } = new();
        public List<float[]> ValidationTargets { get; } = new();
        public List<float> LearningRateHistory { get; } = new();
        public List<DateTime> Timestamps { get; } = new();

        private readonly Queue<float> _recentTrainingLosses = new();
        private readonly Queue<float> _recentValidationLosses = new();
        private const int MaxRecentWindow = 20; // Keep simple like original

        // Enhanced performance tracking but memory-efficient
        public Dictionary<string, List<float>> CustomMetrics { get; } = new();
        public List<long> ParameterCounts { get; } = new();
        public List<int> EpochTimes { get; } = new();

        // Simple rolling calculation - more like original
        private float _validationTrendSum = 0f;
        private int _validationTrendCount = 0;
        private const int TrendWindow = 15;

        public void Log(float trainingLoss, float validationLoss, float[] prediction = null, float[] target = null, float learningRate = 0f, long parameterCount = 0)
        {
            TrainingLosses.Add(trainingLoss);
            ValidationLosses.Add(validationLoss);
            LearningRateHistory.Add(learningRate);
            Timestamps.Add(DateTime.Now);
            ParameterCounts.Add(parameterCount);

            _validationTrendSum += validationLoss;
            _validationTrendCount++;
            if (_validationTrendCount > TrendWindow)
            {
                _validationTrendSum -= ValidationLosses[ValidationLosses.Count - TrendWindow - 1];
                _validationTrendCount = TrendWindow;
            }

            _recentTrainingLosses.Enqueue(trainingLoss);
            _recentValidationLosses.Enqueue(validationLoss);

            if (_recentTrainingLosses.Count > MaxRecentWindow)
            {
                _recentTrainingLosses.Dequeue();
                _recentValidationLosses.Dequeue();
            }

            if (prediction != null && target != null && ValidationPredictions.Count < 20)
            {
                ValidationPredictions.Add((float[])prediction.Clone());
                ValidationTargets.Add((float[])target.Clone());
            }
        }

        public void LogCustomMetric(string name, float value)
        {
            if (!CustomMetrics.ContainsKey(name))
            {
                CustomMetrics[name] = new List<float>();
            }
            CustomMetrics[name].Add(value);

            if (CustomMetrics[name].Count > 100)
            {
                CustomMetrics[name].RemoveAt(0);
            }
        }

        public bool IsOverfitting(int lookback = 8, float threshold = 0.02f)
        {
            if (ValidationLosses.Count < lookback * 2) return false;

            var recentVal = ValidationLosses.TakeLast(lookback).Average();
            var recentTrain = TrainingLosses.TakeLast(lookback).Average();
            var olderVal = ValidationLosses.Skip(ValidationLosses.Count - lookback * 2).Take(lookback).Average();
            var olderTrain = TrainingLosses.Skip(TrainingLosses.Count - lookback * 2).Take(lookback).Average();

            bool valIncreasing = recentVal > olderVal + threshold;
            bool trainDecreasing = recentTrain < olderTrain - threshold / 3;
            bool gapWidening = (recentVal - recentTrain) > (olderVal - olderTrain) + threshold / 2;

            return valIncreasing && trainDecreasing && gapWidening;
        }

        public bool IsPlateaued(float threshold = 5e-5f, int lookback = 12)
        {
            if (ValidationLosses.Count < lookback) return false;

            var recent = ValidationLosses.TakeLast(lookback).ToArray();
            float variance = CalculateVariance(recent);
            float trend = Math.Abs(GetTrend(recent.ToList()));

            return variance < threshold && trend < threshold / 2;
        }

        public bool IsUnstableOutput(float maxErrorSpike = 0.15f, int window = 5)
        {
            if (ValidationLosses.Count < window + 2) return false;

            var recent = ValidationLosses.TakeLast(window).ToArray();
            var differences = recent.Zip(recent.Skip(1), (a, b) => Math.Abs(b - a)).ToArray();

            if (differences.Length == 0) return false;

            float avgDiff = differences.Average();
            float maxDiff = differences.Max();

            return maxDiff > maxErrorSpike && maxDiff > avgDiff * 3;
        }

        public bool HasDeadNeurons(List<Layer> layers, float deadThreshold = 1e-6f, float maxDeadRatio = 0.4f)
        {
            if (layers == null || layers.Count < 2) return false;

            foreach (var layer in layers.Skip(1).Take(layers.Count - 2))
            {
                if (layer.Activations == null || layer.Size == 0)
                {
                    continue;
                }
                var validActivations = layer.Activations.Where(a => !float.IsNaN(a) && !float.IsInfinity(a)).ToArray();
                if (validActivations.Length == 0)
                {
                    continue;
                }
                int deadCount = validActivations.Count(a => Math.Abs(a) < deadThreshold);
                float deadRatio = (float)deadCount / validActivations.Length;

                if (deadRatio > maxDeadRatio)
                {
                    return true;
                }
            }

            return false;
        }

        public bool OutputCollapseDetected(float tolerance = 0.05f, int minSamples = 10)
        {
            if (ValidationPredictions.Count < minSamples)
            {
                return false;
            }

            var recentPredictions = ValidationPredictions.TakeLast(minSamples);
            var allOutputs = recentPredictions.SelectMany(v => v).Where(x => !float.IsNaN(x) && !float.IsInfinity(x)).ToArray();

            if (allOutputs.Length == 0)
            {
                return false;
            }

            float range = allOutputs.Max() - allOutputs.Min();
            float stdDev = CalculateStandardDeviation(allOutputs);

            return range < tolerance && stdDev < tolerance / 2;
        }

        public bool IsDiverging(float threshold = 2.0f, int lookback = 5)
        {
            if (ValidationLosses.Count < lookback + 2)
            {
                return false;
            }

            float currentLoss = ValidationLosses[^1];
            float minRecentLoss = ValidationLosses.TakeLast(lookback).Min();
            float historicalMin = ValidationLosses.Take(Math.Max(1, ValidationLosses.Count - lookback)).Min();

            bool basicDivergence = currentLoss > threshold && currentLoss > historicalMin * 3 && currentLoss > minRecentLoss * 1.5f;

            bool containsNaN = ValidationLosses.TakeLast(lookback).Any(x => float.IsNaN(x) || float.IsInfinity(x));

            return basicDivergence || containsNaN;
        }

        public bool IsConverging(int lookback = 10, float improvementThreshold = 0.01f)
        {
            if (ValidationLosses.Count < lookback * 2)
            {
                return false;
            }

            var recent = ValidationLosses.TakeLast(lookback).Average();
            var older = ValidationLosses.Skip(ValidationLosses.Count - lookback * 2).Take(lookback).Average();

            return (older - recent) > improvementThreshold;
        }

        public float GetValidationTrend(int window = 15)
        {
            if (ValidationLosses.Count < window)
            {
                return 0f;
            }
            return GetTrend(ValidationLosses.TakeLast(window).ToList());
        }

        public float GetTrainingTrend(int window = 15)
        {
            if (TrainingLosses.Count < window)
            {
                return 0f;
            }
            return GetTrend(TrainingLosses.TakeLast(window).ToList());
        }

        private float GetTrend(List<float> values)
        {
            if (values.Count < 3)
            {
                return 0f;
            }

            float n = values.Count;
            float sumX = n * (n - 1) / 2;
            float sumY = values.Sum();
            float sumXY = values.Select((y, i) => i * y).Sum();
            float sumX2 = (n - 1) * n * (2 * n - 1) / 6;

            float denominator = n * sumX2 - sumX * sumX;
            if (Math.Abs(denominator) < 1e-10f)
            {
                return 0f;
            }

            float slope = (n * sumXY - sumX * sumY) / denominator;
            return slope;
        }

        private float CalculateVariance(float[] values)
        {
            if (values.Length < 2)
            {
                return 0f;
            }

            float mean = values.Average();
            return values.Select(x => (x - mean) * (x - mean)).Average();
        }

        private float CalculateStandardDeviation(float[] values)
        {
            return (float)Math.Sqrt(CalculateVariance(values));
        }

        public float LatestValidationLoss => ValidationLosses.Count > 0 ? ValidationLosses[^1] : float.MaxValue;
        public float LatestTrainingLoss => TrainingLosses.Count > 0 ? TrainingLosses[^1] : float.MaxValue;

        public float GetValidationLossImprovement(int window = 10)
        {
            if (ValidationLosses.Count < window * 2) return 0f;

            var recent = ValidationLosses.TakeLast(window).Average();
            var older = ValidationLosses.Skip(Math.Max(0, ValidationLosses.Count - window * 2)).Take(window).Average();

            return older - recent; // Positive means improvement
        }

        public void ClearOldData(int keepRecentCount = 100)
        {
            if (TrainingLosses.Count > keepRecentCount * 2)
            {
                var removeCount = TrainingLosses.Count - keepRecentCount;
                TrainingLosses.RemoveRange(0, removeCount);
                ValidationLosses.RemoveRange(0, removeCount);
                LearningRateHistory.RemoveRange(0, removeCount);
                Timestamps.RemoveRange(0, removeCount);
                ParameterCounts.RemoveRange(0, removeCount);
            }
        }
    }

}
