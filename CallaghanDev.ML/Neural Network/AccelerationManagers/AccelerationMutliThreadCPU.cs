using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AccelerationManagers
{

    public class AccelerationMutliThreadCPU : IAccelerationManager
    {
        private delegate float ActivationFunction(float x);
        private readonly ParallelOptions _parallelOptions;

        public AccelerationMutliThreadCPU()
        {
            _parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };
        }

        public float[] CalculateDotProduct(float[,] matrix, float[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (vector.Length != cols)
                throw new ArgumentException($"Expected vector of length {cols}, got {vector.Length}");

            var result = new float[rows];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++)
                    sum += matrix[i, j] * vector[j];
                result[i] = sum;
            });

            return result;
        }

        public (float[] activation, float[] derivative) ActivateLayer(float[] dot, float[] bias, ActivationType activationType)
        {
            int n = dot.Length;
            var activation = new float[n];
            var derivative = new float[n];

            var func = GetActivationFunction(activationType);
            var deriv = GetActivationFunctionDeriv(activationType);

            Parallel.For(0, n, _parallelOptions, i =>
            {
                float z = dot[i] + bias[i];
                activation[i] = func(z);
                derivative[i] = deriv(z);
            });

            return (activation, derivative);
        }

        public float[] CalculateOutputGradients(float[] cost, float[] derivative)
        {
            int n = cost.Length;
            var grad = new float[n];

            Parallel.For(0, n, _parallelOptions, i =>
                grad[i] = -cost[i] * derivative[i]
            );

            return grad;
        }

        public float[] CalculateHiddenGradients(float[,] weights, float[] nextDeltas, float[] derivative)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var pre = new float[cols];

            // accumulate weighted deltas
            Parallel.For(0, cols, _parallelOptions, j =>
            {
                float sum = 0.0f;
                for (int i = 0; i < rows; i++)
                    sum += weights[i, j] * nextDeltas[i];
                pre[j] = sum;
            });

            var delta = new float[cols];
            Parallel.For(0, cols, _parallelOptions, i =>
                delta[i] = pre[i] * derivative[i]
            );

            return delta;
        }

        public float[,] UpdateWeights(float[,] weights, float[] deltas, float[] prevActivations, float learningRate, float lambda)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var updated = new float[rows, cols];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    float gradStep = deltas[i] * prevActivations[j];
                    float regTerm = lambda * weights[i, j];
                    updated[i, j] = weights[i, j] - learningRate * (gradStep + regTerm);
                }
            });

            return updated;
        }

        public float[] UpdateBias(float[] bias, float[] deltas, float learningRate)
        {
            int n = bias.Length;
            var updated = new float[n];

            Parallel.For(0, n, _parallelOptions, i =>
                updated[i] = bias[i] - learningRate * deltas[i]
            );

            return updated;
        }

        private ActivationFunction GetActivationFunction(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.None: return x => x;
                case ActivationType.Sigmoid: return x => { var k = MathF.Exp(x); return k / (1f + k); };
                case ActivationType.Tanh: return x => MathF.Tanh(x);
                case ActivationType.Relu: return x => x > 0 ? x : 0;
                case ActivationType.Leakyrelu: return x => x > 0 ? x : 0.01f * x;
                default: return x => { var k = MathF.Exp(x); return k / (1f + k); };
            }
        }

        private ActivationFunction GetActivationFunctionDeriv(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.None: return x => 1;
                case ActivationType.Sigmoid: return x => { var e = MathF.Exp(x); var s = e / (1 + e); return s * (1 - s); };
                case ActivationType.Tanh: return x => 1 - MathF.Tanh(x) * MathF.Tanh(x);
                case ActivationType.Relu: return x => x >= 0 ? 1 : 0;
                case ActivationType.Leakyrelu: return x => x >= 0f ? 1f : 0.01f;
                default: return x => { var e = MathF.Exp(x); var s = e / (1 + e); return s * (1 - s); };
            }
        }

        public void Dispose() { }
    }
}