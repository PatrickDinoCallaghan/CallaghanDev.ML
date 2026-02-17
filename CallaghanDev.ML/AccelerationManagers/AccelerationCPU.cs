using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using ILGPU.Algorithms;
using CallaghanDev.ML.Enums;
using System;
using static CallaghanDev.ML.Functions;
using System.Numerics;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationCPU : IAccelerationManager
    {
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

            ActivationFunction activationFunction = Functions.GetActivationFunction(activationType);
            ActivationFunction activationFunctionDeriv = Functions.GetActivationDerivative(activationType);

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

        public float[,] MatrixMultiply(float[,] A, float[,] B)
        {
            int rowsA = A.GetLength(0);
            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);
            int colsB = B.GetLength(1);

            if (colsA != rowsB)
                throw new ArgumentException($"Matrix dimensions don't match: [{rowsA}x{colsA}] * [{rowsB}x{colsB}]");

            var C = new float[rowsA, colsB];

            // Cache-friendly blocked multiplication
            const int BLOCK = 32;
            for (int ii = 0; ii < rowsA; ii += BLOCK)
            {
                for (int jj = 0; jj < colsB; jj += BLOCK)
                {
                    for (int kk = 0; kk < colsA; kk += BLOCK)
                    {
                        int iMax = Math.Min(ii + BLOCK, rowsA);
                        int jMax = Math.Min(jj + BLOCK, colsB);
                        int kMax = Math.Min(kk + BLOCK, colsA);

                        for (int i = ii; i < iMax; i++)
                        {
                            for (int j = jj; j < jMax; j++)
                            {
                                float sum = C[i, j];
                                for (int k = kk; k < kMax; k++)
                                    sum += A[i, k] * B[k, j];
                                C[i, j] = sum;
                            }
                        }
                    }
                }
            }
            return C;
        }

        public float[,] MatrixMultiplyTranspose(float[,] A, float[,] B)
        {
            int rowsA = A.GetLength(0);
            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);
            int colsB = B.GetLength(1);

            if (colsA != colsB)
                throw new ArgumentException($"Matrix dimensions don't match for A*B^T");

            var C = new float[rowsA, rowsB];
            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < rowsB; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < colsA; k++)
                        sum += A[i, k] * B[j, k];
                    C[i, j] = sum;
                }
            }
            return C;
        }

        public float[,] MatrixScale(float[,] matrix, float scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = matrix[i, j] * scalar;
            return result;
        }

        public float[,] MatrixAdd(float[,] A, float[,] B)
        {
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = A[i, j] + B[i, j];
            return result;
        }

        public float[,] Softmax(float[,] matrix, bool[,] mask = null)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    if (mask == null || mask[i, j])
                        max = Math.Max(max, matrix[i, j]);
                }

                float sum = 0.0f;
                for (int j = 0; j < cols; j++)
                {
                    if (mask != null && !mask[i, j])
                        result[i, j] = 0.0f;
                    else
                    {
                        result[i, j] = MathF.Exp(matrix[i, j] - max);
                        sum += result[i, j];
                    }
                }

                for (int j = 0; j < cols; j++)
                    result[i, j] /= sum;
            }
            return result;
        }

        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);
            var result = new float[batchSize, features];

            for (int i = 0; i < batchSize; i++)
            {
                float mean = 0.0f;
                for (int j = 0; j < features; j++)
                    mean += input[i, j];
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
                    result[i, j] = gamma[j] * (input[i, j] - mean) / stdDev + beta[j];
            }
            return result;
        }

        public float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix)
        {
            int outputDim = weights.GetLength(0);
            int inputDim = weights.GetLength(1);
            int seqLen = inputMatrix.GetLength(0);

            if (inputMatrix.GetLength(1) != inputDim)
                throw new ArgumentException($"Expected input columns {inputDim}, got {inputMatrix.GetLength(1)}");

            var result = new float[seqLen, outputDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < inputDim; k++)
                    {
                        sum += weights[j, k] * inputMatrix[i, k];
                    }
                    result[i, j] = sum;
                }
            }
            return result;
        }

        #region Multi-head attention

        public float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);  // K and V have the same seq length
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            var concatenated = new float[seqLenQ, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                var Q_head = new float[seqLenQ, headDim];
                var K_head = new float[seqLenK, headDim];
                var V_head = new float[seqLenK, headDim];

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        Q_head[i, j] = Q[i, startIdx + j];
                    }
                }

                for (int i = 0; i < seqLenK; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        K_head[i, j] = K[i, startIdx + j];
                        V_head[i, j] = V[i, startIdx + j];
                    }
                }

                var scores = MatrixMultiplyTranspose(Q_head, K_head);
                var scaledScores = MatrixScale(scores, scale);
                var attnWeights = Softmax(scaledScores, mask);

                var headOutput = MatrixMultiply(attnWeights, V_head);

                for (int i = 0; i < seqLenQ; i++)
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
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            var dQ_full = new float[seqLenQ, embeddingDim];
            var dK_full = new float[seqLenK, embeddingDim];
            var dV_full = new float[seqLenK, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                var Q_head = new float[seqLenQ, headDim];
                var dHeadOutput = new float[seqLenQ, headDim];

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        Q_head[i, j] = Q[i, startIdx + j];
                        dHeadOutput[i, j] = dConcatenated[i, startIdx + j];
                    }
                }

                var K_head = new float[seqLenK, headDim];
                var V_head = new float[seqLenK, headDim];

                for (int i = 0; i < seqLenK; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        K_head[i, j] = K[i, startIdx + j];
                        V_head[i, j] = V[i, startIdx + j];
                    }
                }

                // Recompute attention weights: scores [seqLenQ, seqLenK]
                var scores = MatrixMultiplyTranspose(Q_head, K_head);

                var scaledScores = MatrixScale(scores, scale);

                // Softmax with optional causal mask: attnWeights [seqLenQ, seqLenK]
                var attnWeights = new float[seqLenQ, seqLenK];

                for (int i = 0; i < seqLenQ; i++)
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (useDecoderMask && j > i) continue;
                        max = Math.Max(max, scaledScores[i, j]);
                    }
                    float expSum = 0;
                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (useDecoderMask && j > i)
                        {
                            attnWeights[i, j] = 0;
                            continue;
                        }
                        attnWeights[i, j] = MathF.Exp(scaledScores[i, j] - max);
                        expSum += attnWeights[i, j];
                    }
                    for (int j = 0; j < seqLenK; j++)
                    {
                        attnWeights[i, j] /= (expSum + 1e-10f);
                    }
                }

                // dAttnWeights[i,j] = sum_k dHeadOutput[i,k] * V_head[j,k]
                // Shape: [seqLenQ, seqLenK]
                var dAttnWeights = new float[seqLenQ, seqLenK];

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < seqLenK; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < headDim; k++)
                        {
                            sum += dHeadOutput[i, k] * V_head[j, k];
                        }
                        dAttnWeights[i, j] = sum;
                    }
                }

                // dV_head[j,k] = sum_i attnWeights[i,j] * dHeadOutput[i,k]
                // Shape: [seqLenK, headDim]
                var dV_head = new float[seqLenK, headDim];

                for (int j = 0; j < seqLenK; j++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0;
                        for (int i = 0; i < seqLenQ; i++)
                        {
                            sum += attnWeights[i, j] * dHeadOutput[i, k];
                        }
                        dV_head[j, k] = sum;
                    }
                }

                // Softmax backward: dScaledScores [seqLenQ, seqLenK]
                var dScaledScores = new float[seqLenQ, seqLenK];

                for (int i = 0; i < seqLenQ; i++)
                {
                    float dot = 0;
                    for (int j = 0; j < seqLenK; j++)
                    {
                        dot += attnWeights[i, j] * dAttnWeights[i, j];
                    }
                    for (int j = 0; j < seqLenK; j++)
                    {
                        dScaledScores[i, j] = attnWeights[i, j] * (dAttnWeights[i, j] - dot);
                        if (useDecoderMask && j > i)
                        {
                            dScaledScores[i, j] = 0;
                        }
                    }
                }

                var dScores = new float[seqLenQ, seqLenK];

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < seqLenK; j++)
                    {
                        dScores[i, j] = dScaledScores[i, j] * scale;
                    }
                }

                // dQ_head[i,k] = sum_j dScores[i,j] * K_head[j,k]
                // Shape: [seqLenQ, headDim]
                var dQ_head = new float[seqLenQ, headDim];
                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqLenK; j++)
                        {
                            sum += dScores[i, j] * K_head[j, k];
                        }
                        dQ_head[i, k] = sum;
                    }
                }

                // dK_head[j,k] = sum_i dScores[i,j] * Q_head[i,k]
                // Shape: [seqLenK, headDim]
                var dK_head = new float[seqLenK, headDim];
                for (int j = 0; j < seqLenK; j++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0;
                        for (int i = 0; i < seqLenQ; i++)
                        {
                            sum += dScores[i, j] * Q_head[i, k];
                        }
                        dK_head[j, k] = sum;
                    }
                }

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        dQ_full[i, startIdx + j] += dQ_head[i, j];
                    }
                }

                for (int i = 0; i < seqLenK; i++)
                    for (int j = 0; j < headDim; j++)
                    {
                        dK_full[i, startIdx + j] += dK_head[i, j];
                        dV_full[i, startIdx + j] += dV_head[i, j];
                    }
            }

            return (dQ_full, dK_full, dV_full);
        }

        #endregion

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

            for (int i = 0; i < batchSize; i++)
            {
                float mean = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    mean += input[i, j];
                }
                mean /= features;
                means[i] = mean;

                float variance = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    float diff = input[i, j] - mean;
                    variance += diff * diff;
                }
                variance = variance / features;

                variances[i] = variance;

                float stdDev = MathF.Sqrt(variance + epsilon);
                for (int j = 0; j < features; j++)
                {
                    normalized[i, j] = (input[i, j] - mean) / stdDev;
                    output[i, j] = gamma[j] * normalized[i, j] + beta[j];
                }
            }

            return (output, means, variances, normalized);
        }

        public (float[,] dInput, float[] dGamma, float[] dBeta) LayerNormBackward(float[,] dOut, float[,] normalized, float[] gamma, float[,] input, float[] mean, float[] variance, float epsilon = 1e-5f)
        {
            int batchSize = dOut.GetLength(0);
            int features = dOut.GetLength(1);

            var dInput = new float[batchSize, features];
            var dGamma = new float[features];
            var dBeta = new float[features];

            for (int i = 0; i < batchSize; i++)
            {
                float invStd = 1.0f / MathF.Sqrt(variance[i] + epsilon);

                for (int j = 0; j < features; j++)
                {
                    dGamma[j] += dOut[i, j] * normalized[i, j];
                    dBeta[j] += dOut[i, j];
                }

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
            }

            return (dInput, dGamma, dBeta);
        }

        public float MatrixSquaredNorm(float[,] matrix)
        {
            float sum = 0;
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    sum += matrix[i, j] * matrix[i, j];
                }
            }
            return sum;
        }

        public void MatrixScaleInPlace(float[,] matrix, float scale)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] *= scale;
                }
            }
        }

        public void VectorScaleInPlace(float[] vector, float scale)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] *= scale;
            }
        }

        public void MatrixUpdate(float[,] weights, float[,] gradients, float learningRate)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    weights[i, j] -= learningRate * gradients[i, j];
                }
            }
        }

        public void VectorUpdate(float[] weights, float[] gradients, float learningRate)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] -= learningRate * gradients[i];
            }
        }

        public void ZeroMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = 0;
                }
            }
        }


        //new
        public float[,] MatrixAddBias(float[,] matrix, float[] bias)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] + bias[j];
                }
            }
            return result;
        }

        public float[,] EmbedTokensWithPosition(float[,] tokenEmbedding, int[] tokenIds, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            var result = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                int tokenId = tokenIds[i];
                for (int j = 0; j < embeddingDim; j++)
                {
                    result[i, j] = tokenEmbedding[tokenId, j] + positionalEncoding[i, j];
                }
            }
            return result;
        }

        public float[,] AddBiasAndPositionalEncoding(float[,] projected, float[] bias, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            var result = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    result[i, j] = projected[i, j] + bias[j] + positionalEncoding[i, j];
                }
            }
            return result;
        }

        public (float loss, float[,] dLogits) CrossEntropyLossAndGradient(float[,] logits, int[] targets, int effectiveLen)
        {
            int outputDim = logits.GetLength(1);
            float loss = 0;
            var dLogits = new float[logits.GetLength(0), outputDim];
            float invLen = 1.0f / effectiveLen;

            for (int i = 0; i < effectiveLen; i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < outputDim; j++)
                {
                    max = Math.Max(max, logits[i, j]);
                }

                float sum = 0;
                var probs = new float[outputDim];
                for (int j = 0; j < outputDim; j++)
                {
                    probs[j] = MathF.Exp(logits[i, j] - max);
                    sum += probs[j];
                }
                for (int j = 0; j < outputDim; j++)
                {
                    probs[j] /= sum;
                }

                int targetToken = targets[i];
                loss -= MathF.Log(probs[targetToken] + 1e-10f);

                for (int j = 0; j < outputDim; j++)
                {
                    dLogits[i, j] = probs[j] * invLen;
                    if (j == targetToken)
                    {
                        dLogits[i, j] -= invLen;
                    }
                }
            }

            loss /= effectiveLen;
            return (loss, dLogits);
        }

        public (float loss, float[,] dOutput) MSELossAndGradient(float[,] predictions, float[,] targets, int effectiveLen)
        {
            int outputDim = predictions.GetLength(1);
            float loss = 0;
            var dOutput = new float[predictions.GetLength(0), outputDim];
            float invLen = 1.0f / (effectiveLen * outputDim);

            for (int i = 0; i < effectiveLen; i++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float diff = predictions[i, j] - targets[i, j];
                    loss += diff * diff;
                    dOutput[i, j] = 2.0f * diff * invLen;
                }
            }

            loss /= (effectiveLen * outputDim);
            return (loss, dOutput);
        }

        public float[,] BackpropOutputProjection(float[,] dLogits, float[,] input, float[,] weights, float[,] weightGrad, float[] biasGrad, int seqLen, int outputDim, int embeddingDim)
        {
            var dX = new float[seqLen, embeddingDim];

            for (int i = 0; i < seqLen; i++)
            {
                for (int v = 0; v < outputDim; v++)
                {
                    float dVal = dLogits[i, v];
                    for (int e = 0; e < embeddingDim; e++)
                    {
                        weightGrad[v, e] += input[i, e] * dVal;
                    }
                    biasGrad[v] += dVal;
                }

                for (int e = 0; e < embeddingDim; e++)
                {
                    float grad = 0;
                    for (int v = 0; v < outputDim; v++)
                    {
                        grad += dLogits[i, v] * weights[v, e];
                    }
                    dX[i, e] = grad;
                }
            }

            return dX;
        }

        public void BackpropInputProjection(float[,] dX, float[,] continuousInput, float[,] weightGrad, float[] biasGrad, int seqLen, int embeddingDim, int inputFeatureDim)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int e = 0; e < embeddingDim; e++)
                {
                    float dVal = dX[i, e];
                    for (int f = 0; f < inputFeatureDim; f++)
                    {
                        weightGrad[e, f] += dVal * continuousInput[i, f];
                    }
                    biasGrad[e] += dVal;
                }
            }
        }

        public void AccumulateTokenEmbeddingGrad(float[,] embeddingGrad, float[,] dX, int[] tokenIds, int seqLen, int embeddingDim)
        {
            for (int i = 0; i < seqLen; i++)
            {
                int tokenId = tokenIds[i];
                for (int j = 0; j < embeddingDim; j++)
                {
                    embeddingGrad[tokenId, j] += dX[i, j];
                }
            }
        }

        public void AccumulateVectorGradients(float[] targetGrad, float[] sourceGrad)
        {
            for (int j = 0; j < targetGrad.Length; j++)
            {
                targetGrad[j] += sourceGrad[j];
            }
        }

        public float VectorSquaredNorm(float[] vector)
        {
            float sum = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += vector[i] * vector[i];
            }
            return sum;
        }

        public float[,] SliceRows(float[,] matrix, int startRow, int endRow)
        {
            int cols = matrix.GetLength(1);
            int numRows = endRow - startRow;
            var result = new float[numRows, cols];
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[startRow + i, j];
                }
            }
            return result;
        }

        public float[] ExtractRow(float[,] matrix, int rowIndex, int cols)
        {
            var result = new float[cols];
            for (int j = 0; j < cols; j++)
            {
                result[j] = matrix[rowIndex, j];
            }
            return result;
        }

        public void SetRow(float[,] matrix, int rowIndex, float[] values, int cols)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[rowIndex, j] = values[j];
            }
        }

        public bool[,] CreateCausalMask(int seqLen)
        {
            var mask = new bool[seqLen, seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    mask[i, j] = true;
                }
            }
            return mask;
        }

        public void MatrixAccumulate(float[,] target, float[,] source)
        {
            int rows = target.GetLength(0);
            int cols = target.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    target[i, j] += source[i, j];
                }
            }
        }

        public void ZeroVector(float[] vector)
        {
            Array.Clear(vector, 0, vector.Length);
        }

        public void SigmoidInPlace(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float x = matrix[i, j];
                    if (x >= 0)
                    {
                        float ex = MathF.Exp(-x);
                        matrix[i, j] = 1.0f / (1.0f + ex);
                    }
                    else
                    {
                        float ex = MathF.Exp(x);
                        matrix[i, j] = ex / (1.0f + ex);
                    }
                }
            }
        }
        public void Dispose() { }

    }
}
