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

            var func = Functions.GetActivationFunction(activationType);
            var deriv = Functions.GetActivationDerivative(activationType);

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
                {
                    sum += weights[i, j] * nextDeltas[i];
                }
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

        public float[,] MatrixMultiply(float[,] A, float[,] B)
        {
            int rowsA = A.GetLength(0);
            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);
            int colsB = B.GetLength(1);

            if (colsA != rowsB)
            {
                throw new ArgumentException($"Matrix dimensions don't match");
            }

            var C = new float[rowsA, colsB];
            const int BLOCK = 32;

            Parallel.For(0, (rowsA + BLOCK - 1) / BLOCK, _parallelOptions, ii =>
            {
                int iStart = ii * BLOCK;
                int iEnd = Math.Min(iStart + BLOCK, rowsA);

                for (int jj = 0; jj < colsB; jj += BLOCK)
                {
                    int jEnd = Math.Min(jj + BLOCK, colsB);
                    for (int kk = 0; kk < colsA; kk += BLOCK)
                    {
                        int kEnd = Math.Min(kk + BLOCK, colsA);
                        for (int i = iStart; i < iEnd; i++)
                        {
                            for (int j = jj; j < jEnd; j++)
                            {
                                float sum = C[i, j];
                                for (int k = kk; k < kEnd; k++)
                                {
                                    sum += A[i, k] * B[k, j];
                                }
                                C[i, j] = sum;
                            }
                        }
                    }
                }
            });
            return C;
        }

        public float[,] MatrixMultiplyTranspose(float[,] A, float[,] B)
        {
            int rowsA = A.GetLength(0);
            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);
            int colsB = B.GetLength(1);

            if (colsA != colsB)
            {
                throw new ArgumentException($"Matrix dimensions don't match for A*B^T");
            }

            var C = new float[rowsA, rowsB];
            Parallel.For(0, rowsA, _parallelOptions, i =>
            {
                for (int j = 0; j < rowsB; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < colsA; k++)
                    {
                        sum += A[i, k] * B[j, k];
                    }
                    C[i, j] = sum;
                }
            });
            return C;
        }

        public float[,] MatrixScale(float[,] matrix, float scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] * scalar;
                }
            });
            return result;
        }

        public float[,] MatrixAdd(float[,] A, float[,] B)
        {
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);
            var result = new float[rows, cols];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = A[i, j] + B[i, j];
                }
            });
            return result;
        }

        public float[,] Softmax(float[,] matrix, bool[,] mask = null)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    if (mask == null || mask[i, j])
                    {
                        max = Math.Max(max, matrix[i, j]);
                    }
                }

                float sum = 0.0f;
                for (int j = 0; j < cols; j++)
                {
                    if (mask != null && !mask[i, j])
                    {
                        result[i, j] = 0.0f;
                    }
                    else
                    {
                        result[i, j] = MathF.Exp(matrix[i, j] - max);
                        sum += result[i, j];
                    }
                }

                for (int j = 0; j < cols; j++)
                {
                    result[i, j] /= sum;
                }
            });
            return result;
        }

        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);
            var result = new float[batchSize, features];

            Parallel.For(0, batchSize, _parallelOptions, i =>
            {
                float mean = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    mean += input[i, j];
                }
                mean /= features;

                float variance = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    float diff = input[i, j] - mean;
                    variance += diff * diff;
                }
                variance /= features;

                float stdDev = MathF.Sqrt(variance + epsilon);
                for (int j = 0; j < features; j++)
                {
                    result[i, j] = gamma[j] * (input[i, j] - mean) / stdDev + beta[j];
                }
            });
            return result;
        }
        public void Dispose() { }
    }
}