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
        private delegate double ActivationFunction(double x);
        private readonly ParallelOptions _parallelOptions;

        public AccelerationMutliThreadCPU()
        {
            _parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };
        }

        public double[] CalculateDotProduct(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (vector.Length != cols)
                throw new ArgumentException($"Expected vector of length {cols}, got {vector.Length}");

            var result = new double[rows];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                double sum = 0.0;
                for (int j = 0; j < cols; j++)
                    sum += matrix[i, j] * vector[j];
                result[i] = sum;
            });

            return result;
        }

        public (double[] activation, double[] derivative) ActivateLayer(double[] dot, double[] bias, ActivationType activationType)
        {
            int n = dot.Length;
            var activation = new double[n];
            var derivative = new double[n];

            var func = GetActivationFunction(activationType);
            var deriv = GetActivationFunctionDeriv(activationType);

            Parallel.For(0, n, _parallelOptions, i =>
            {
                double z = dot[i] + bias[i];
                activation[i] = func(z);
                derivative[i] = deriv(z);
            });

            return (activation, derivative);
        }

        public double[] CalculateOutputGradients(double[] cost, double[] derivative)
        {
            int n = cost.Length;
            var grad = new double[n];

            Parallel.For(0, n, _parallelOptions, i =>
                grad[i] = -cost[i] * derivative[i]
            );

            return grad;
        }

        public double[] CalculateHiddenGradients(double[,] weights, double[] nextDeltas, double[] derivative)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var pre = new double[cols];

            // accumulate weighted deltas
            Parallel.For(0, cols, _parallelOptions, j =>
            {
                double sum = 0.0;
                for (int i = 0; i < rows; i++)
                    sum += weights[i, j] * nextDeltas[i];
                pre[j] = sum;
            });

            var delta = new double[cols];
            Parallel.For(0, cols, _parallelOptions, i =>
                delta[i] = pre[i] * derivative[i]
            );

            return delta;
        }

        public double[,] UpdateWeights(double[,] weights, double[] deltas, double[] prevActivations, double learningRate, double lambda)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var updated = new double[rows, cols];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    double gradStep = deltas[i] * prevActivations[j];
                    double regTerm = lambda * weights[i, j];
                    updated[i, j] = weights[i, j] - learningRate * (gradStep + regTerm);
                }
            });

            return updated;
        }

        public double[] UpdateBias(double[] bias, double[] deltas, double learningRate)
        {
            int n = bias.Length;
            var updated = new double[n];

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
                case ActivationType.Sigmoid: return x => { var k = Math.Exp(x); return k / (1 + k); };
                case ActivationType.Tanh: return x => Math.Tanh(x);
                case ActivationType.Relu: return x => x > 0 ? x : 0;
                case ActivationType.Leakyrelu: return x => x > 0 ? x : 0.01 * x;
                default: return x => { var k = Math.Exp(x); return k / (1 + k); };
            }
        }

        private ActivationFunction GetActivationFunctionDeriv(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.None: return x => 1;
                case ActivationType.Sigmoid: return x => { var e = Math.Exp(x); var s = e / (1 + e); return s * (1 - s); };
                case ActivationType.Tanh: return x => 1 - Math.Tanh(x) * Math.Tanh(x);
                case ActivationType.Relu: return x => x >= 0 ? 1 : 0;
                case ActivationType.Leakyrelu: return x => x >= 0 ? 1 : 0.01;
                default: return x => { var e = Math.Exp(x); var s = e / (1 + e); return s * (1 - s); };
            }
        }
    }
}