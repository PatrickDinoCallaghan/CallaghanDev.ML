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
    
        public float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix)
        {
            int outputDim = weights.GetLength(0);
            int inputDim = weights.GetLength(1);
            int seqLen = inputMatrix.GetLength(0);

            if (inputMatrix.GetLength(1) != inputDim)
            {

                throw new ArgumentException($"Expected input columns {inputDim}, got {inputMatrix.GetLength(1)}");

            }
            var result = new float[seqLen, outputDim];
            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < inputDim; k++)
                        sum += weights[j, k] * inputMatrix[i, k];
                    result[i, j] = sum;
                }
            });
            return result;
        }

        public float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            int seqLen = Q.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            var concatenated = new float[seqLen, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                var Q_head = new float[seqLen, headDim];
                var K_head = new float[seqLen, headDim];
                var V_head = new float[seqLen, headDim];

                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        Q_head[i, j] = Q[i, startIdx + j];
                        K_head[i, j] = K[i, startIdx + j];
                        V_head[i, j] = V[i, startIdx + j];
                    }
                }

                var scores = MatrixMultiplyTranspose(Q_head, K_head);
                var scaledScores = MatrixScale(scores, scale);
                var attnWeights = Softmax(scaledScores, mask);
                var headOutput = MatrixMultiply(attnWeights, V_head);

                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        concatenated[i, startIdx + j] = headOutput[i, j];
                    }
                }
            }

            return concatenated;
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            int seqLen = Q.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            var dQ_full = new float[seqLen, embeddingDim];
            var dK_full = new float[seqLen, embeddingDim];
            var dV_full = new float[seqLen, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                var Q_head = new float[seqLen, headDim];
                var K_head = new float[seqLen, headDim];
                var V_head = new float[seqLen, headDim];
                var dHeadOutput = new float[seqLen, headDim];

                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        Q_head[i, j] = Q[i, startIdx + j];
                        K_head[i, j] = K[i, startIdx + j];
                        V_head[i, j] = V[i, startIdx + j];
                        dHeadOutput[i, j] = dConcatenated[i, startIdx + j];
                    }
                }

                var scores = MatrixMultiplyTranspose(Q_head, K_head);
                var scaledScores = MatrixScale(scores, scale);

                var attnWeights = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (useDecoderMask && j > i)
                        {
                            continue;
                        }
                        max = Math.Max(max, scaledScores[i, j]);
                    }
                    float expSum = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (useDecoderMask && j > i)
                        {
                            attnWeights[i, j] = 0;
                            continue;
                        }
                        attnWeights[i, j] = MathF.Exp(scaledScores[i, j] - max);
                        expSum += attnWeights[i, j];
                    }
                    for (int j = 0; j < seqLen; j++)
                    {
                        attnWeights[i, j] /= (expSum + 1e-10f);
                    }
                }

                var dAttnWeights = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < headDim; k++)
                        {
                            sum += dHeadOutput[i, k] * V_head[j, k];
                        }
                        dAttnWeights[i, j] = sum;
                    }
                }

                var dV_head = new float[seqLen, headDim];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqLen; j++)
                        {
                            sum += attnWeights[j, i] * dHeadOutput[j, k];
                        }
                        dV_head[i, k] = sum;
                    }
                }

                var dScaledScores = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    float dot = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        dot += attnWeights[i, j] * dAttnWeights[i, j];
                    }
                    for (int j = 0; j < seqLen; j++)
                    {
                        dScaledScores[i, j] = attnWeights[i, j] * (dAttnWeights[i, j] - dot);
                        if (useDecoderMask && j > i)
                        {
                            dScaledScores[i, j] = 0;
                        }
                    }
                }

                var dScores = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        dScores[i, j] = dScaledScores[i, j] * scale;
                    }
                }

                var dQ_head = new float[seqLen, headDim];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqLen; j++)
                        {

                            sum += dScores[i, j] * K_head[j, k];
                        }
                        dQ_head[i, k] = sum;
                    }
                }

                var dK_head = new float[seqLen, headDim];
                for (int j = 0; j < seqLen; j++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0;
                        for (int i = 0; i < seqLen; i++)
                            sum += dScores[i, j] * Q_head[i, k];
                        dK_head[j, k] = sum;
                    }
                }

                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        dQ_full[i, startIdx + j] += dQ_head[i, j];
                        dK_full[i, startIdx + j] += dK_head[i, j];
                        dV_full[i, startIdx + j] += dV_head[i, j];
                    }
                }
            }

            return (dQ_full, dK_full, dV_full);
        }

        public void Dispose() { }
        public void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput)
        {
            int seqLen = input.GetLength(0);
            int embeddingDim = input.GetLength(1);

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    float dOutVal = dOutput[i, j];

                    for (int k = 0; k < embeddingDim; k++)
                    {
                        weightGrad[k, j] += input[i, k] * dOutVal;
                    }

                    biasGrad[j] += dOutVal;
                }

                for (int k = 0; k < embeddingDim; k++)
                {
                    float sum = 0;
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        sum += dOutput[i, j] * weights[k, j];
                    }
                    dInput[i, k] += sum;
                }
            }
        }

        public (float[,] output, float[] means, float[] variances, float[,] normalized) LayerNormForward(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            var means = new float[batchSize];
            var variances = new float[batchSize];
            var normalized = new float[batchSize, features];
            var output = new float[batchSize, features];

            Parallel.For(0, batchSize, _parallelOptions, i =>
            {
                float mean = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    mean += input[i, j];
                }
                mean = mean/ features;
                means[i] = mean;

                float variance = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    float diff = input[i, j] - mean;
                    variance += diff * diff;
                }
                variance = variance/ features;
                variances[i] = variance;

                float stdDev = MathF.Sqrt(variance + epsilon);
                for (int j = 0; j < features; j++)
                {
                    normalized[i, j] = (input[i, j] - mean) / stdDev;
                    output[i, j] = gamma[j] * normalized[i, j] + beta[j];
                }
            });

            return (output, means, variances, normalized);
        }

        public (float[,] dInput, float[] dGamma, float[] dBeta) LayerNormBackward(float[,] dOut, float[,] normalized, float[] gamma, float[,] input, float[] mean, float[] variance, float epsilon = 1e-5f)
        {
            int batchSize = dOut.GetLength(0);
            int features = dOut.GetLength(1);

            var dInput = new float[batchSize, features];
            var dGamma = new float[features];
            var dBeta = new float[features];


            var localDGammas = new float[batchSize][];
            var localDBetas = new float[batchSize][];

            Parallel.For(0, batchSize, _parallelOptions, i =>
            {
                float invStd = 1.0f / MathF.Sqrt(variance[i] + epsilon);

                var ldGamma = new float[features];
                var ldBeta = new float[features];

                for (int j = 0; j < features; j++)
                {
                    ldGamma[j] = dOut[i, j] * normalized[i, j];
                    ldBeta[j] = dOut[i, j];
                }

                localDGammas[i] = ldGamma;
                localDBetas[i] = ldBeta;

                var dNorm = new float[features];
                for (int j = 0; j < features; j++)
                {

                    dNorm[j] = dOut[i, j] * gamma[j];

                }
                float dVar = 0;
                float invStdCubed = invStd * invStd * invStd;
                for (int j = 0; j < features; j++)
                {
                    float xMinusMean = input[i, j] - mean[i];
                    dVar += dNorm[j] * xMinusMean * (-0.5f) * invStdCubed;
                }

                float dMean = 0;
                for (int j = 0; j < features; j++)
                {
                    dMean += dNorm[j] * (-invStd);
                }

                float invN = 1.0f / features;
                for (int j = 0; j < features; j++)
                {
                    float xMinusMean = input[i, j] - mean[i];
                    dInput[i, j] = dNorm[j] * invStd + dVar * 2.0f * xMinusMean * invN + dMean * invN;
                }
            });

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    dGamma[j] += localDGammas[i][j];
                    dBeta[j] += localDBetas[i][j];
                }
            }

            return (dInput, dGamma, dBeta);
        }

        public float MatrixSquaredNorm(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float sum = 0;
            object lockObj = new object();

            Parallel.For(0, rows, _parallelOptions, () => 0.0f,
                (i, state, localSum) =>
                {
                    for (int j = 0; j < cols; j++)
                        localSum += matrix[i, j] * matrix[i, j];
                    return localSum;
                },
                localSum => { lock (lockObj) { sum += localSum; } }
            );

            return sum;
        }

        public void MatrixScaleInPlace(float[,] matrix, float scale)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                    matrix[i, j] *= scale;
            });
        }

        public void VectorScaleInPlace(float[] vector, float scale)
        {
            Parallel.For(0, vector.Length, _parallelOptions, i =>
                vector[i] *= scale
            );
        }

        public void MatrixUpdate(float[,] weights, float[,] gradients, float learningRate)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                    weights[i, j] -= learningRate * gradients[i, j];
            });
        }

        public void VectorUpdate(float[] weights, float[] gradients, float learningRate)
        {
            Parallel.For(0, weights.Length, _parallelOptions, i =>
                weights[i] -= learningRate * gradients[i]
            );
        }

        public void ZeroMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = 0;
            });
        }
    }
}