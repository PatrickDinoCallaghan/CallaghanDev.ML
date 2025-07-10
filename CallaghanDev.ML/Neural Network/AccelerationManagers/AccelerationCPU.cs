using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using ILGPU.Algorithms;
using CallaghanDev.ML.Enums;
using System;

namespace CallaghanDev.ML.AccelerationManagers
{

    public class AccelerationCPU : IAccelerationManager
    {
        private delegate double ActivationFunction(double x);

        public AccelerationCPU()
        {
        }
        
        // I realize this behaves like a transpose rather than a plain dot product—but that’s exactly what we want, and it’s correct.
        public double[] CalculateDotProduct(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0);   // number of neurons in this layer
            int cols = matrix.GetLength(1);   // number of neurons in previous layer
            if (vector.Length != cols)
            {
                throw new ArgumentException($"Expected vector of length {cols}, got {vector.Length}");
            }

            var result = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < cols; j++)
                {
                    sum += matrix[i, j] * vector[j];
                }
                result[i] = sum;
            }
            return result;
        }
        public (double[] activation, double[] derivative) ActivateLayer(double[] dot, double[] bias, ActivationType activationType)
        {
            int n = dot.Length;
            var activation = new double[n];
            var derivative = new double[n];

            ActivationFunction activationFunction = GetActivationFunction(activationType);
            ActivationFunction activationFunctionDeriv = GetActivationFunctionDeriv(activationType);

            for (int i = 0; i < n; i++)
            {
                double z = dot[i] + bias[i];
                activation[i] = activationFunction(z);
                derivative[i] = activationFunctionDeriv(z);
            }
            return (activation, derivative);
        }

        public double[] CalculateOutputGradients(double[] cost, double[] derivative)
        {
            int n = cost.Length;
            var grad = new double[n];
            for (int i = 0; i < n; i++)
            {
                grad[i] = -cost[i] * derivative[i];
            }
            return grad;
        }

        public double[] CalculateHiddenGradients(double[,] weights, double[] nextDeltas, double[] derivative)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var pre = new double[cols];

            for (int j = 0; j < cols; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < rows; i++)
                    sum += weights[i, j] * nextDeltas[i];
                pre[j] = sum;
            }

            var delta = new double[cols];
            for (int i = 0; i < cols; i++)
            {
                delta[i] = pre[i] * derivative[i];
            }

            return delta;
        }

        public double[,] UpdateWeights(double[,] weights,double[] deltas, double[] prevActivations, double learningRate, double lambda )
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var updated = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double gradStep = deltas[i] * prevActivations[j];

                    double regTerm = lambda * weights[i, j];

                    updated[i, j] = weights[i, j]
                                  - learningRate * (gradStep + regTerm);
                }
            }
            return updated;
        }

        public double[] UpdateBias(double[] bias, double[] deltas, double learningRate)
        {
            int n = bias.Length;
            var updated = new double[n];
            for (int i = 0; i < n; i++)
            {
                updated[i] = bias[i] - learningRate * deltas[i];
            }
            return updated;
        }


        private ActivationFunction GetActivationFunction(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.None:
                    return x => x;
                case ActivationType.Sigmoid:
                    return x =>
                    {
                        double k = Math.Exp(x);
                        return k / (1.0d + k);
                    };
                case ActivationType.Tanh:
                    return x =>
                    {
                        return Math.Tanh(x);
                    };
                case ActivationType.Relu:
                    return x =>
                    {
                        return 0 >= x ? 0 : x;
                    };
                case ActivationType.Leakyrelu:
                    return x =>
                    {
                        return 0 >= x ? 0.01d * x : x;
                    };
                default: // ActivationType.Sigmoid:
                    return x =>
                    {
                        double k = Math.Exp(x);
                        return k / (1.0d + k);
                    };
            }
        }

        private ActivationFunction GetActivationFunctionDeriv(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.None:

                    return x => 1;

                case ActivationType.Sigmoid:
                    return x =>
                    {
                        return Math.Exp(x) / (1.0d + Math.Exp(x)) * (1 - Math.Exp(x) / (1.0d + Math.Exp(x)));
                    };
                case ActivationType.Tanh:
                    return x =>
                    {
                        return 1 - x * x;
                    };
                case ActivationType.Relu:
                    return x =>
                    {
                        return x >= 0 ? 1 : 0;
                    };
                case ActivationType.Leakyrelu:
                    return x =>
                    {
                        return 0 >= x ? 0.1d : 1;
                    };
                default: //ActivationType.Sigmoid:
                    return x =>
                    {
                        return Math.Exp(x) / (1.0d + Math.Exp(x)) * (1 - Math.Exp(x) / (1.0d + Math.Exp(x)));
                    };
            }
        }
    }
}
