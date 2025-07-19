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
        private delegate float ActivationFunction(float x);

        public AccelerationCPU()
        {
        }
        
        // I realize this behaves like a transpose rather than a plain dot product—but that’s exactly what we want, and it’s correct.
        public float[] CalculateDotProduct(float[,] matrix, float[] vector)
        {
            int rows = matrix.GetLength(0);   // number of neurons in this layer
            int cols = matrix.GetLength(1);   // number of neurons in previous layer
            if (vector.Length != cols)
            {
                throw new ArgumentException($"Expected vector of length {cols}, got {vector.Length}");
            }

            var result = new float[rows];
            for (int i = 0; i < rows; i++)
            {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++)
                {
                    sum += matrix[i, j] * vector[j];
                }
                result[i] = sum;
            }
            return result;
        }
        public (float[] activation, float[] derivative) ActivateLayer(float[] dot, float[] bias, ActivationType activationType)
        {
            int n = dot.Length;
            var activation = new float[n];
            var derivative = new float[n];

            ActivationFunction activationFunction = GetActivationFunction(activationType);
            ActivationFunction activationFunctionDeriv = GetActivationFunctionDeriv(activationType);

            for (int i = 0; i < n; i++)
            {
                float z = dot[i] + bias[i];
                activation[i] = activationFunction(z);
                derivative[i] = activationFunctionDeriv(z);
            }
            return (activation, derivative);
        }

        public float[] CalculateOutputGradients(float[] cost, float[] derivative)
        {
            int n = cost.Length;
            var grad = new float[n];
            for (int i = 0; i < n; i++)
            {
                grad[i] = -cost[i] * derivative[i];
            }
            return grad;
        }

        public float[] CalculateHiddenGradients(float[,] weights, float[] nextDeltas, float[] derivative)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var pre = new float[cols];

            for (int j = 0; j < cols; j++)
            {
                float sum = 0.0f;
                for (int i = 0; i < rows; i++)
                {
                    sum += weights[i, j] * nextDeltas[i];
                }
                pre[j] = sum;
            }

            var delta = new float[cols];
            for (int i = 0; i < cols; i++)
            {
                delta[i] = pre[i] * derivative[i];
            }

            return delta;
        }

        public float[,] UpdateWeights(float[,] weights,float[] deltas, float[] prevActivations, float learningRate, float lambda )
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var updated = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float gradStep = deltas[i] * prevActivations[j];

                    float regTerm = lambda * weights[i, j];

                    updated[i, j] = weights[i, j]
                                  - learningRate * (gradStep + regTerm);
                }
            }
            return updated;
        }

        public float[] UpdateBias(float[] bias, float[] deltas, float learningRate)
        {
            int n = bias.Length;
            var updated = new float[n];
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
                        float k = MathF.Exp(x);
                        return k / (1.0f + k);
                    };
                case ActivationType.Tanh:
                    return x =>
                    {
                        return MathF.Tanh(x);
                    };
                case ActivationType.Relu:
                    return x =>
                    {
                        return 0 >= x ? 0 : x;
                    };
                case ActivationType.Leakyrelu:
                    return x =>
                    {
                        return 0 >= x ? 0.01f * x : x;
                    };
                default: // ActivationType.Sigmoid:
                    return x =>
                    {
                        float k = MathF.Exp(x);
                        return k / (1.0f + k);
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
                        return MathF.Exp(x) / (1.0f + MathF.Exp(x)) * (1 - MathF.Exp(x) / (1.0f+ MathF.Exp(x)));
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
                        return 0 >= x ? 0.1f : 1;
                    };
                default: //ActivationType.Sigmoid:
                    return x =>
                    {
                        return MathF.Exp(x) / (1.0f + MathF.Exp(x)) * (1 - MathF.Exp(x) / (1.0f + MathF.Exp(x)));
                    };
            }
        }

        public void Dispose() { }
    }
    
}
