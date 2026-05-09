using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.TACAMT;
using ILGPU.Algorithms;
using MathNet.Numerics;
using System.Buffers;
using System.Text;
using static CallaghanDev.ML.Functions;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationCPU : IAccelerationManager
    {
        public AccelerationCPU()
        {
        }

        #region Shared Tensor primitives

        public float[,] MatrixMultiply(float[,] A, float[,] B)
        {
            int rowsA = A.GetLength(0);
            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);
            int colsB = B.GetLength(1);

            if (colsA != rowsB)
            {
                throw new ArgumentException($"Matrix dimensions don't match: [{rowsA}x{colsA}] * [{rowsB}x{colsB}]");
            }

            var C = new float[rowsA, colsB];

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
                                {
                                    sum += A[i, k] * B[k, j];
                                }
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
            {
                throw new ArgumentException($"Matrix dimensions don't match for A*B^T");
            }

            var C = new float[rowsA, rowsB];
            for (int i = 0; i < rowsA; i++)
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
            }
            return C;
        }

        public float[,] MatrixScale(float[,] matrix, float scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] * scalar;
                }
            }
            return result;
        }

        public float[,] MatrixAdd(float[,] A, float[,] B)
        {
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = A[i, j] + B[i, j];
                }
            }
            return result;
        }

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

        public float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix)
        {
            int seqLen = inputMatrix.GetLength(0);

            // send to offset-aware version
            return BatchDotProduct(weights, inputMatrix, rowStart: 0, rowCount: seqLen);
        }

        public float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix, int rowStart, int rowCount)
        {
            if (weights == null)
            {
                throw new ArgumentNullException(nameof(weights));
            }
            if (inputMatrix == null)
            {
                throw new ArgumentNullException(nameof(inputMatrix));
            } 

            if (rowStart < 0 || rowCount < 0)
            {
                throw new ArgumentOutOfRangeException();
            }

            if (rowStart + rowCount > inputMatrix.GetLength(0))
            {
                throw new ArgumentException("Invalid row slice.");
            }

            int outputDim = weights.GetLength(0);
            int inputDim = weights.GetLength(1);

            if (inputMatrix.GetLength(1) != inputDim)
            {
                throw new ArgumentException($"Expected input columns {inputDim}, got {inputMatrix.GetLength(1)}");
            }

            var result = new float[rowCount, outputDim];

            for (int i = 0; i < rowCount; i++)
            {
                int srcRow = rowStart + i;

                for (int j = 0; j < outputDim; j++)
                {
                    float sum = 0.0f;

                    for (int k = 0; k < inputDim; k++)
                    {
                        sum += weights[j, k] * inputMatrix[srcRow, k];
                    }

                    result[i, j] = sum;
                }
            }

            return result;
        }

        public float[,] SliceRows(float[,] matrix, int startRow, int endRow)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            if (startRow < 0 || endRow > matrix.GetLength(0) || startRow > endRow)
            {
                throw new ArgumentOutOfRangeException();
            }

            int cols = matrix.GetLength(1);
            int numRows = endRow - startRow;
            var result = new float[numRows, cols];

            int bytesPerRow = cols * sizeof(float);

            for (int i = 0; i < numRows; i++)
            {
                Buffer.BlockCopy(matrix, (startRow + i) * bytesPerRow, result, i * bytesPerRow, bytesPerRow);
            }

            return result;
        }

        public float[] ExtractRow(float[,] matrix, int rowIndex, int cols)
        {
            var result = new float[cols];

            Buffer.BlockCopy(matrix, rowIndex * cols * sizeof(float), result, 0, cols * sizeof(float));

            return result;
        }

        public void SetRow(float[,] matrix, int rowIndex, float[] values, int cols)
        {
            Buffer.BlockCopy(values, 0, matrix, rowIndex * cols * sizeof(float), cols * sizeof(float));
        }

        public void ZeroMatrix(float[,] matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            Array.Clear(matrix, 0, matrix.Length);
        }

        public void ZeroVector(float[] vector)
        {
            Array.Clear(vector, 0, vector.Length);
        }

        public void MatrixAddInPlace(float[,] target, float[,] addend)
        {
            int rows = target.GetLength(0);
            int cols = target.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    target[i, j] += addend[i, j];
                }
            }
        }

        public void VectorAccumulate(float[] target, float[] source)
        {
            for (int j = 0; j < target.Length; j++)
            {
                target[j] += source[j];
            }
        }

        #endregion

        #region Neural network
        // I realize this behaves like a transpose rather than a plain dot product-but that's exactly what we want, and it's correct.
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
            if (bias.Length != dot.Length)
            {
                throw new ArgumentException("Bias length must match dot product length");
            }

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

        public float[,] UpdateWeights(float[,] weights, float[] deltas, float[] prevActivations, float learningRate, float lambda)
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

                    updated[i, j] = weights[i, j] - learningRate * (gradStep + regTerm);
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

        #endregion

        #region Transformer core

        // Replace Softmax and MultiHeadAttentionBackward inside AccelerationCPU.
        // Add the bool[,] overload to IAccelerationManager too, or keep the bool wrapper for existing callers.

        public float[,] Softmax(float[,] matrix, bool[,] mask = null)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (mask != null && (mask.GetLength(0) != rows || mask.GetLength(1) != cols))
            {
                throw new ArgumentException($"Mask shape must be [{rows},{cols}], got [{mask.GetLength(0)},{mask.GetLength(1)}].", nameof(mask));
            }

            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                float max = float.NegativeInfinity;

                for (int j = 0; j < cols; j++)
                {
                    if (mask != null && !mask[i, j])
                    {
                        continue;
                    }

                    float value = matrix[i, j];
                    if (float.IsNaN(value))
                    {
                        throw new InvalidOperationException($"Softmax input contains NaN at [{i},{j}].");
                    }

                    if (value > max)
                    {
                        max = value;
                    }
                }

                if (float.IsNegativeInfinity(max))
                {
                    continue; // Entire row is masked. Return zeros for that row.
                }

                float sum = 0.0f;
                for (int j = 0; j < cols; j++)
                {
                    if (mask != null && !mask[i, j])
                    {
                        result[i, j] = 0.0f;
                        continue;
                    }

                    float exp = MathF.Exp(matrix[i, j] - max);
                    result[i, j] = exp;
                    sum += exp;
                }

                if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
                {
                    continue;
                }

                float invSum = 1.0f / sum;
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] *= invSum;
                }
            }

            return result;
        }

        #region MultiHeadAttentionBackward

        [Obsolete]
        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward_Obsolete(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            bool[,] mask = null;

            if (useDecoderMask)
            {
                int seqLenQ = Q.GetLength(0);
                int seqLenK = K.GetLength(0);
                mask = new bool[seqLenQ, seqLenK];

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < seqLenK; j++)
                    {
                        mask[i, j] = j <= i;
                    }
                }
            }

            return MultiHeadAttentionBackward(Q, K, V, dConcatenated, numHeads, scale, mask);
        }

        [Obsolete]
        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward_Obsolete(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask)
        {
            if (Q == null) throw new ArgumentNullException(nameof(Q));
            if (K == null) throw new ArgumentNullException(nameof(K));
            if (V == null) throw new ArgumentNullException(nameof(V));
            if (dConcatenated == null) throw new ArgumentNullException(nameof(dConcatenated));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));

            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);

            if (K.GetLength(1) != embeddingDim || V.GetLength(1) != embeddingDim)
                throw new ArgumentException("Q, K and V must have the same embedding dimension.");
            if (V.GetLength(0) != seqLenK)
                throw new ArgumentException("K and V must have the same sequence length.");
            if (dConcatenated.GetLength(0) != seqLenQ || dConcatenated.GetLength(1) != embeddingDim)
                throw new ArgumentException("dConcatenated shape must match Q shape.", nameof(dConcatenated));
            if (embeddingDim % numHeads != 0)
                throw new ArgumentException("Embedding dim must be divisible by numHeads.", nameof(numHeads));
            if (mask != null && (mask.GetLength(0) != seqLenQ || mask.GetLength(1) != seqLenK))
                throw new ArgumentException($"Mask shape must be [{seqLenQ},{seqLenK}], got [{mask.GetLength(0)},{mask.GetLength(1)}].", nameof(mask));

            int headDim = embeddingDim / numHeads;
            var dQFull = new float[seqLenQ, embeddingDim];
            var dKFull = new float[seqLenK, embeddingDim];
            var dVFull = new float[seqLenK, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int offset = head * headDim;

                var attnWeights = new float[seqLenQ, seqLenK];

                for (int i = 0; i < seqLenQ; i++)
                {
                    float maxScore = float.NegativeInfinity;
                    var scaledScores = new float[seqLenK];

                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (mask != null && !mask[i, j])
                        {
                            scaledScores[j] = float.NegativeInfinity;
                            continue;
                        }

                        float dot = 0f;
                        for (int k = 0; k < headDim; k++)
                            dot += Q[i, offset + k] * K[j, offset + k];

                        float score = dot * scale;
                        scaledScores[j] = score;
                        if (score > maxScore) maxScore = score;
                    }

                    if (float.IsNegativeInfinity(maxScore))
                        continue;

                    float sumExp = 0f;
                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (float.IsNegativeInfinity(scaledScores[j]))
                        {
                            attnWeights[i, j] = 0f;
                            continue;
                        }

                        float w = MathF.Exp(scaledScores[j] - maxScore);
                        attnWeights[i, j] = w;
                        sumExp += w;
                    }

                    if (sumExp > 0f)
                    {
                        float inv = 1f / sumExp;
                        for (int j = 0; j < seqLenK; j++)
                            attnWeights[i, j] *= inv;
                    }
                }

                var dAttnWeights = new float[seqLenQ, seqLenK];
                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < seqLenK; j++)
                    {
                        float sum = 0f;
                        for (int k = 0; k < headDim; k++)
                            sum += dConcatenated[i, offset + k] * V[j, offset + k];
                        dAttnWeights[i, j] = sum;
                    }
                }

                for (int j = 0; j < seqLenK; j++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0f;
                        for (int i = 0; i < seqLenQ; i++)
                            sum += attnWeights[i, j] * dConcatenated[i, offset + k];
                        dVFull[j, offset + k] += sum;
                    }
                }

                var dDot = new float[seqLenQ, seqLenK];
                for (int i = 0; i < seqLenQ; i++)
                {
                    float rowDot = 0f;
                    for (int j = 0; j < seqLenK; j++)
                        rowDot += attnWeights[i, j] * dAttnWeights[i, j];

                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (mask != null && !mask[i, j])
                        {
                            dDot[i, j] = 0f;
                            continue;
                        }

                        float dScaledScore = attnWeights[i, j] * (dAttnWeights[i, j] - rowDot);
                        dDot[i, j] = dScaledScore * scale;
                    }
                }

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0f;
                        for (int j = 0; j < seqLenK; j++)
                            sum += dDot[i, j] * K[j, offset + k];
                        dQFull[i, offset + k] += sum;
                    }
                }

                for (int j = 0; j < seqLenK; j++)
                {
                    for (int k = 0; k < headDim; k++)
                    {
                        float sum = 0f;
                        for (int i = 0; i < seqLenQ; i++)
                            sum += dDot[i, j] * Q[i, offset + k];
                        dKFull[j, offset + k] += sum;
                    }
                }
            }

            return (dQFull, dKFull, dVFull);
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            return MultiHeadAttentionBackwardCore(
                Q,
                K,
                V,
                dConcatenated,
                numHeads,
                scale,
                mask: null,
                useDecoderMask: useDecoderMask);
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask)
        {
            return MultiHeadAttentionBackwardCore(
                Q,
                K,
                V,
                dConcatenated,
                numHeads,
                scale,
                mask,
                useDecoderMask: false);
        }

        private (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackwardCore(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask, bool useDecoderMask)
        {
            if (Q == null) throw new ArgumentNullException(nameof(Q));
            if (K == null) throw new ArgumentNullException(nameof(K));
            if (V == null) throw new ArgumentNullException(nameof(V));
            if (dConcatenated == null) throw new ArgumentNullException(nameof(dConcatenated));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));

            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);

            if (K.GetLength(1) != embeddingDim || V.GetLength(1) != embeddingDim)
                throw new ArgumentException("Q, K and V must have the same embedding dimension.");

            if (V.GetLength(0) != seqLenK)
                throw new ArgumentException("K and V must have the same sequence length.");

            if (dConcatenated.GetLength(0) != seqLenQ || dConcatenated.GetLength(1) != embeddingDim)
                throw new ArgumentException("dConcatenated shape must match Q shape.", nameof(dConcatenated));

            if (embeddingDim % numHeads != 0)
                throw new ArgumentException("Embedding dim must be divisible by numHeads.", nameof(numHeads));

            if (mask != null && (mask.GetLength(0) != seqLenQ || mask.GetLength(1) != seqLenK))
                throw new ArgumentException(
                    $"Mask shape must be [{seqLenQ},{seqLenK}], got [{mask.GetLength(0)},{mask.GetLength(1)}].",
                    nameof(mask));

            int headDim = embeddingDim / numHeads;

            var dQFull = new float[seqLenQ, embeddingDim];
            var dKFull = new float[seqLenK, embeddingDim];
            var dVFull = new float[seqLenK, embeddingDim];

            // Reused scratch buffers.
            var weights = new float[seqLenK];
            var dAttn = new float[seqLenK];
            var activeIndices = new int[seqLenK];

            var qRow = new float[headDim];
            var doutRow = new float[headDim];
            var dqRow = new float[headDim];

            for (int head = 0; head < numHeads; head++)
            {
                int offset = head * headDim;

                for (int i = 0; i < seqLenQ; i++)
                {
                    Array.Clear(dqRow, 0, headDim);

                    for (int k = 0; k < headDim; k++)
                    {
                        int col = offset + k;
                        qRow[k] = Q[i, col];
                        doutRow[k] = dConcatenated[i, col];
                    }

                    float maxScore = float.NegativeInfinity;
                    int activeCount = 0;

                    if (mask != null)
                    {
                        for (int j = 0; j < seqLenK; j++)
                        {
                            if (!mask[i, j])
                                continue;

                            float dot = 0f;

                            for (int k = 0; k < headDim; k++)
                                dot += qRow[k] * K[j, offset + k];

                            float score = dot * scale;

                            weights[j] = score;
                            activeIndices[activeCount++] = j;

                            if (score > maxScore)
                                maxScore = score;
                        }
                    }
                    else
                    {
                        int visibleKeys = useDecoderMask
                            ? Math.Min(i + 1, seqLenK)
                            : seqLenK;

                        activeCount = visibleKeys;

                        for (int j = 0; j < visibleKeys; j++)
                        {
                            float dot = 0f;

                            for (int k = 0; k < headDim; k++)
                                dot += qRow[k] * K[j, offset + k];

                            float score = dot * scale;

                            weights[j] = score;

                            if (score > maxScore)
                                maxScore = score;
                        }
                    }

                    if (activeCount == 0 || float.IsNegativeInfinity(maxScore))
                        continue;

                    float sumExp = 0f;

                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];

                            float w = MathF.Exp(weights[j] - maxScore);
                            weights[j] = w;
                            sumExp += w;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < activeCount; j++)
                        {
                            float w = MathF.Exp(weights[j] - maxScore);
                            weights[j] = w;
                            sumExp += w;
                        }
                    }

                    if (sumExp <= 0f)
                        continue;

                    float invSumExp = 1f / sumExp;

                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];
                            weights[j] *= invSumExp;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < activeCount; j++)
                            weights[j] *= invSumExp;
                    }

                    float rowDot = 0f;

                    // Compute dV and dAttn row.
                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];
                            float w = weights[j];

                            float dAttnJ = 0f;

                            for (int k = 0; k < headDim; k++)
                            {
                                int col = offset + k;
                                float dout = doutRow[k];

                                dVFull[j, col] += w * dout;
                                dAttnJ += dout * V[j, col];
                            }

                            dAttn[j] = dAttnJ;
                            rowDot += w * dAttnJ;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < activeCount; j++)
                        {
                            float w = weights[j];

                            float dAttnJ = 0f;

                            for (int k = 0; k < headDim; k++)
                            {
                                int col = offset + k;
                                float dout = doutRow[k];

                                dVFull[j, col] += w * dout;
                                dAttnJ += dout * V[j, col];
                            }

                            dAttn[j] = dAttnJ;
                            rowDot += w * dAttnJ;
                        }
                    }

                    // Softmax backward + Q/K backward.
                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];

                            float dDot = weights[j] * (dAttn[j] - rowDot) * scale;

                            if (dDot == 0f)
                                continue;

                            for (int k = 0; k < headDim; k++)
                            {
                                int col = offset + k;

                                dqRow[k] += dDot * K[j, col];
                                dKFull[j, col] += dDot * qRow[k];
                            }
                        }
                    }
                    else
                    {
                        for (int j = 0; j < activeCount; j++)
                        {
                            float dDot = weights[j] * (dAttn[j] - rowDot) * scale;

                            if (dDot == 0f)
                                continue;

                            for (int k = 0; k < headDim; k++)
                            {
                                int col = offset + k;

                                dqRow[k] += dDot * K[j, col];
                                dKFull[j, col] += dDot * qRow[k];
                            }
                        }
                    }

                    for (int k = 0; k < headDim; k++)
                        dQFull[i, offset + k] += dqRow[k];
                }
            }

            return (dQFull, dKFull, dVFull);
        }

        #endregion

        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);
            var result = new float[batchSize, features];

            for (int i = 0; i < batchSize; i++)
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
            }
            return result;
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

        public bool[,] CreateCausalMask(int seqLen)
        {
            if (seqLen < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(seqLen));
            }

            var mask = new bool[seqLen, seqLen];

            var trueRow = new bool[seqLen];
            Array.Fill(trueRow, true);

            const int boolSize = sizeof(bool); // 1 byte

            for (int i = 0; i < seqLen; i++)
            {
                Buffer.BlockCopy(trueRow, 0, mask, i * seqLen * boolSize, (i + 1) * boolSize);
            }

            return mask;
        }

        #region MultiHeadAttentionForward

        [Obsolete]
        public float[,] MultiHeadAttentionForward_Obsolete(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);  // K and V have the same seq length
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads");
            }
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

        public float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            if (Q == null)
            {
                throw new ArgumentNullException(nameof(Q));
            }
            if (K == null)
            {
                throw new ArgumentNullException(nameof(K));
            }
            if (V == null)
            {
                throw new ArgumentNullException(nameof(V));
            } 

            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);

            if (K.GetLength(1) != embeddingDim || V.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("Q, K and V must have the same embedding dimension.");
            }

            if (V.GetLength(0) != seqLenK)
            {
                throw new ArgumentException("K and V must have the same sequence length.");
            }

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads.");
            }

            if (mask != null && (mask.GetLength(0) != seqLenQ || mask.GetLength(1) != seqLenK))
            {
                throw new ArgumentException("Mask shape must be [seqLenQ, seqLenK].");
            }

            int headDim = embeddingDim / numHeads;
            var output = new float[seqLenQ, embeddingDim];

            float[] scores = ArrayPool<float>.Shared.Rent(seqLenK);

            try
            {
                for (int head = 0; head < numHeads; head++)
                {
                    int startIdx = head * headDim;

                    for (int i = 0; i < seqLenQ; i++)
                    {
                        float maxScore = float.NegativeInfinity;

                        // Compute QK^T row directly for this query token.
                        for (int k = 0; k < seqLenK; k++)
                        {
                            if (mask != null && !mask[i, k])
                            {
                                scores[k] = float.NegativeInfinity;
                                continue;
                            }

                            float dot = 0f;

                            for (int d = 0; d < headDim; d++)
                            {
                                dot += Q[i, startIdx + d] * K[k, startIdx + d];
                            }

                            float score = dot * scale;
                            scores[k] = score;

                            if (score > maxScore)
                            {
                                maxScore = score;
                            }
                        }

                        // Handle fully masked row.
                        if (float.IsNegativeInfinity(maxScore))
                        {
                            for (int d = 0; d < headDim; d++)
                            {
                                output[i, startIdx + d] = 0f;
                            }

                            continue;
                        }

                        // Stable softmax.
                        float sumExp = 0f;

                        for (int k = 0; k < seqLenK; k++)
                        {
                            if (float.IsNegativeInfinity(scores[k]))
                            {
                                scores[k] = 0f;
                                continue;
                            }

                            float exp = MathF.Exp(scores[k] - maxScore);
                            scores[k] = exp;
                            sumExp += exp;
                        }

                        float invSumExp = sumExp > 0f ? 1f / sumExp : 0f;

                        for (int k = 0; k < seqLenK; k++)
                        {
                            scores[k] *= invSumExp;
                        }

                        // Multiply attention row by V directly into output.
                        for (int d = 0; d < headDim; d++)
                        {
                            float value = 0f;

                            for (int k = 0; k < seqLenK; k++)
                            {
                                value += scores[k] * V[k, startIdx + d];
                            }

                            output[i, startIdx + d] = value;
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(scores);
            }

            return output;
        }

        #endregion

        /*
        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;
            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads");
            }
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
                        //attnWeights[i, j] /= (expSum + 1e-10f);
                        attnWeights[i, j] = attnWeights[i, j] / expSum;
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
        */
        public float[,] FFNForwardBatch(float[,] input, int seqLen, int outputDim, Func<float[], float[]> forwardPassFn)
        {
            var result = new float[seqLen, outputDim];

            for (int i = 0; i < seqLen; i++)
            {
                var row = new float[input.GetLength(1)];

                for (int j = 0; j < input.GetLength(1); j++)
                {
                    row[j] = input[i, j];
                }

                var out_row = forwardPassFn(row);

                for (int j = 0; j < outputDim; j++)
                {
                    result[i, j] = out_row[j];
                }
            }

            return result;
        }

        #endregion

        #region Transformer training
        public void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput)
        {
            int seqLen = input.GetLength(0);
            int inDim = input.GetLength(1);
            int outDim = dOutput.GetLength(1);

            for (int i = 0; i < seqLen; i++)
            {
                // weight + bias grads
                for (int j = 0; j < outDim; j++)
                {
                    float dOutVal = dOutput[i, j];

                    for (int k = 0; k < inDim; k++)
                    {
                        weightGrad[j, k] += dOutVal * input[i, k];
                    }

                    biasGrad[j] += dOutVal;
                }

                // input grad
                for (int k = 0; k < inDim; k++)
                {
                    float sum = 0;

                    for (int j = 0; j < outDim; j++)
                    {
                        sum += dOutput[i, j] * weights[j, k];
                    }

                    dInput[i, k] += sum;
                }
            }
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

        public float VectorSquaredNorm(float[] vector)
        {
            float sum = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += vector[i] * vector[i];
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

        #endregion

        #region Multimodal/TACAMT/MMTAC

        public void ApplyContextTypeEmbedding(float[,] contextHidden, float[,] typeEmbedding, int[] typeIndices)
        {
            if (contextHidden == null)
                throw new ArgumentNullException(nameof(contextHidden));
            if (typeEmbedding == null)
                throw new ArgumentNullException(nameof(typeEmbedding));
            if (typeIndices == null)
                throw new ArgumentNullException(nameof(typeIndices));

            int rows = contextHidden.GetLength(0);
            int embDim = contextHidden.GetLength(1);

            if (typeIndices.Length != rows)
                throw new ArgumentException("typeIndices length must match contextHidden row count.", nameof(typeIndices));
            if (typeEmbedding.GetLength(1) != embDim)
                throw new ArgumentException("typeEmbedding dimension must match contextHidden embedding dimension.", nameof(typeEmbedding));

            int typeCount = typeEmbedding.GetLength(0);

            for (int i = 0; i < rows; i++)
            {
                int type = typeIndices[i];
                if ((uint)type >= (uint)typeCount)
                    throw new ArgumentOutOfRangeException(nameof(typeIndices), $"Invalid context type index {type} at row {i}.");

                for (int d = 0; d < embDim; d++)
                    contextHidden[i, d] += typeEmbedding[type, d];
            }
        }

        public float[,] ComputeTimeDiffMatrix(int priceSeqLen, float[] keyArrivalTimes)
        {
            if (priceSeqLen <= 0)
                throw new ArgumentOutOfRangeException(nameof(priceSeqLen), "priceSeqLen must be positive.");
            if (keyArrivalTimes == null)
                throw new ArgumentNullException(nameof(keyArrivalTimes));

            int numKeys = keyArrivalTimes.Length;
            var td = new float[priceSeqLen, numKeys];

            for (int p = 0; p < priceSeqLen; p++)
            {
                for (int s = 0; s < numKeys; s++)
                {
                    // Positive means the key is in the past relative to query position p.
                    // Future keys are clamped to zero inside ContentAwareDecayForward.
                    td[p, s] = p - keyArrivalTimes[s];
                }
            }

            return td;
        }

        public float[] ComputeMemoryAttentionScores(float[,] priceHidden, int lastPos, float[,] contextHidden, int totalCtx, float scale)
        {
            int embDim = priceHidden.GetLength(1);
            var scores = new float[totalCtx];

            for (int s = 0; s < totalCtx; s++)
            {
                float dot = 0f;
                for (int d = 0; d < embDim; d++)
                    dot += priceHidden[lastPos, d] * contextHidden[s, d];
                scores[s] = dot * scale;
            }

            return scores;
        }

        public float[,] ProjectOutputBatch(float[,] hidden, float[,] outputProjection, float[] outputBias, int seqLen, int outputDim)
        {
            int embDim = hidden.GetLength(1);
            var pred = new float[seqLen, outputDim];

            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < outputDim; j++)
                {
                    float sum = outputBias[j];
                    for (int k = 0; k < embDim; k++)
                        sum += outputProjection[j, k] * hidden[i, k];
                    pred[i, j] = sum;
                }

            return pred;
        }

        public (float[,,] decayBias, ContentAwareDecayCache cache) ContentAwareDecayForward(float[,] queryEmbeddings, float[,] keyEmbeddings, float[,] timeDiffs, float[] keyTimesFromRef, ContentAwareDecayNetwork network, bool isTraining = false, Random dropoutRng = null)
        {
            if (queryEmbeddings == null) throw new ArgumentNullException(nameof(queryEmbeddings));
            if (keyEmbeddings == null) throw new ArgumentNullException(nameof(keyEmbeddings));
            if (timeDiffs == null) throw new ArgumentNullException(nameof(timeDiffs));
            if (network == null) throw new ArgumentNullException(nameof(network));

            int queryLen = timeDiffs.GetLength(0);
            int keyLen = timeDiffs.GetLength(1);
            int numHeads = network.NumHeads;
            int projDim = network.ProjectionDim;
            int contentDim = network.ContentDim;
            int hiddenDim = network.HiddenDim;
            int mlpInputDim = network.MLPInputDim;
            int numBases = network.NumTimeBases;
            int rawDim = network.TimeRawDim;

            if (queryEmbeddings.GetLength(0) != queryLen || queryEmbeddings.GetLength(1) != contentDim)
                throw new ArgumentException("queryEmbeddings shape does not match the decay network.", nameof(queryEmbeddings));
            if (keyEmbeddings.GetLength(0) != keyLen || keyEmbeddings.GetLength(1) != contentDim)
                throw new ArgumentException("keyEmbeddings shape does not match the decay network.", nameof(keyEmbeddings));
            if (keyTimesFromRef != null && keyTimesFromRef.Length != keyLen)
                throw new ArgumentException("keyTimesFromRef length must match key length.", nameof(keyTimesFromRef));

            var normalizedTimeDiffs = new float[queryLen, keyLen];
            float timeNorm = MathF.Max(network.TimeNormalizationHours, 1e-4f);

            for (int qi = 0; qi < queryLen; qi++)
            {
                for (int si = 0; si < keyLen; si++)
                {
                    float td = timeDiffs[qi, si];
                    normalizedTimeDiffs[qi, si] = td > 0f ? td / timeNorm : 0f;
                }
            }

            var cache = new ContentAwareDecayCache
            {
                QueryEmbeddings = queryEmbeddings,
                KeyEmbeddings = keyEmbeddings,
                TimeDiffs = timeDiffs,
                NormalizedTimeDiffs = normalizedTimeDiffs,
                KeyTimesFromRef = keyTimesFromRef,
                QueryProj = new float[numHeads, queryLen, projDim],
                KeyProj = new float[numHeads, keyLen, projDim],
                TimeRawFeatures = new float[numHeads, keyLen, rawDim],
                TimeEncoding = new float[numHeads, keyLen, projDim],
                MemAttnQInput = new float[numHeads, keyLen, projDim],
                MemAttnKInput = new float[numHeads, keyLen, projDim],
                MemAttnWeights = new float[numHeads, keyLen, keyLen],
                MemAttnOutput = new float[numHeads, keyLen, projDim],
                RefinedKey = new float[numHeads, keyLen, projDim],
                MLPInput = new float[queryLen, keyLen, numHeads, mlpInputDim],
                MLPHiddenPreAct = new float[queryLen, keyLen, numHeads, hiddenDim],
                MLPHidden = new float[queryLen, keyLen, numHeads, hiddenDim],
                GateLogits = new float[queryLen, keyLen, numHeads],
                Gates = new float[queryLen, keyLen, numHeads],
                MemAttnDropoutMask = null,
                MLPDropoutMask = null
            };

            bool useMemAttnDrop = isTraining && network.MemoryAttentionDropout > 0f && dropoutRng != null;
            bool useMLPDrop = isTraining && network.MLPDropout > 0f && dropoutRng != null;

            if (useMemAttnDrop)
            {
                if (network.MemoryAttentionDropout >= 1f)
                    throw new ArgumentOutOfRangeException(nameof(network.MemoryAttentionDropout), "MemoryAttentionDropout must be < 1.");

                cache.MemAttnDropoutMask = new float[numHeads, keyLen, keyLen];
            }

            if (useMLPDrop)
            {
                if (network.MLPDropout >= 1f)
                    throw new ArgumentOutOfRangeException(nameof(network.MLPDropout), "MLPDropout must be < 1.");

                cache.MLPDropoutMask = new float[queryLen, keyLen, numHeads, hiddenDim];
            }

            var decayBias = new float[queryLen, keyLen, numHeads];

            for (int h = 0; h < numHeads; h++)
            {
                for (int q = 0; q < queryLen; q++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.QueryProjectionBias[h, p];
                        for (int d = 0; d < contentDim; d++)
                            val += network.QueryProjection[h, p, d] * queryEmbeddings[q, d];

                        cache.QueryProj[h, q, p] = val;
                    }
                }

                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.KeyProjectionBias[h, p];
                        for (int d = 0; d < contentDim; d++)
                            val += network.KeyProjection[h, p, d] * keyEmbeddings[s, d];

                        cache.KeyProj[h, s, p] = val;
                    }
                }

                for (int s = 0; s < keyLen; s++)
                {
                    float t = keyTimesFromRef != null ? keyTimesFromRef[s] : 0f;

                    for (int b = 0; b < numBases; b++)
                    {
                        float freq = MathF.Exp(network.TimeLogFreq[h, b]);
                        float angle = freq * t;
                        cache.TimeRawFeatures[h, s, b * 2] = MathF.Sin(angle);
                        cache.TimeRawFeatures[h, s, b * 2 + 1] = MathF.Cos(angle);
                    }

                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.TimeProjBias[h, p];
                        for (int r = 0; r < rawDim; r++)
                            val += network.TimeProj[h, p, r] * cache.TimeRawFeatures[h, s, r];

                        cache.TimeEncoding[h, s, p] = val;
                    }
                }

                float memScale = 1.0f / MathF.Sqrt(projDim);

                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float kp = cache.KeyProj[h, s, p] + cache.TimeEncoding[h, s, p];
                        cache.MemAttnQInput[h, s, p] = kp;
                        cache.MemAttnKInput[h, s, p] = kp;
                    }
                }

                bool useMemAttentionTemporalMask = keyTimesFromRef != null;

                for (int i = 0; i < keyLen; i++)
                {
                    float maxScore = float.NegativeInfinity;
                    var scores = new float[keyLen];
                    float queryKeyTime = useMemAttentionTemporalMask ? keyTimesFromRef[i] : 0f;
                    for (int j = 0; j < keyLen; j++)
                    {
                        bool valid =
                            keyTimesFromRef == null ||
                            keyTimesFromRef[j] <= keyTimesFromRef[i];

                        if (!valid)
                        {
                            scores[j] = float.NegativeInfinity;
                            continue;
                        }

                        float dot = 0f;
                        for (int p = 0; p < projDim; p++)
                            dot += cache.MemAttnQInput[h, i, p] * cache.MemAttnKInput[h, j, p];

                        scores[j] = dot * memScale;

                        if (scores[j] > maxScore)
                            maxScore = scores[j];
                    }
                    if (float.IsNegativeInfinity(maxScore))
                        continue;

                    float sumExp = 0f;
                    for (int j = 0; j < keyLen; j++)
                    {
                        if (float.IsNegativeInfinity(scores[j]))
                        {
                            cache.MemAttnWeights[h, i, j] = 0f;
                            continue;
                        }

                        float w = MathF.Exp(scores[j] - maxScore);
                        cache.MemAttnWeights[h, i, j] = w;
                        sumExp += w;
                    }

                    if (sumExp > 0f)
                    {
                        float inv = 1f / sumExp;
                        for (int j = 0; j < keyLen; j++)
                            cache.MemAttnWeights[h, i, j] *= inv;
                    }

                    if (useMemAttnDrop)
                    {
                        float keepProb = 1.0f - network.MemoryAttentionDropout;
                        float scaleDrop = 1.0f / keepProb;

                        for (int j = 0; j < keyLen; j++)
                        {
                            if (cache.MemAttnWeights[h, i, j] == 0f)
                            {
                                cache.MemAttnDropoutMask[h, i, j] = 0f;
                                continue;
                            }

                            float mask = dropoutRng.NextSingle() < keepProb ? scaleDrop : 0f;
                            cache.MemAttnDropoutMask[h, i, j] = mask;
                            cache.MemAttnWeights[h, i, j] *= mask;
                        }
                    }

                    for (int p = 0; p < projDim; p++)
                    {
                        float val = 0f;
                        for (int j = 0; j < keyLen; j++)
                            val += cache.MemAttnWeights[h, i, j] * cache.KeyProj[h, j, p];

                        cache.MemAttnOutput[h, i, p] = val;
                    }
                }

                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.MemAttnOutputB[h, p];
                        for (int q = 0; q < projDim; q++)
                            val += network.MemAttnOutputW[h, p, q] * cache.MemAttnOutput[h, s, q];

                        cache.RefinedKey[h, s, p] = val + cache.KeyProj[h, s, p];
                    }
                }

                float baseRate = MathF.Exp(network.LogBaseDecayRate[h]);

                for (int qi = 0; qi < queryLen; qi++)
                {
                    for (int si = 0; si < keyLen; si++)
                    {
                        if (timeDiffs[qi, si] < 0f)
                        {
                            cache.Gates[qi, si, h] = 0f;
                            cache.GateLogits[qi, si, h] = 0f;
                            decayBias[qi, si, h] = float.NegativeInfinity;
                            continue;
                        }

                        float normTd = normalizedTimeDiffs[qi, si];
                        float logTd = MathF.Log(1f + normTd);

                        int idx = 0;
                        for (int p = 0; p < projDim; p++)
                            cache.MLPInput[qi, si, h, idx++] = cache.QueryProj[h, qi, p];

                        for (int p = 0; p < projDim; p++)
                            cache.MLPInput[qi, si, h, idx++] = cache.RefinedKey[h, si, p];

                        for (int p = 0; p < projDim; p++)
                            cache.MLPInput[qi, si, h, idx++] = cache.QueryProj[h, qi, p] * cache.RefinedKey[h, si, p];

                        cache.MLPInput[qi, si, h, idx++] = normTd;
                        cache.MLPInput[qi, si, h, idx++] = logTd;

                        for (int j = 0; j < hiddenDim; j++)
                        {
                            float val = network.B1[h, j];
                            for (int k = 0; k < mlpInputDim; k++)
                                val += network.W1[h, j, k] * cache.MLPInput[qi, si, h, k];

                            cache.MLPHiddenPreAct[qi, si, h, j] = val;
                            float activated = val > 0f ? val : 0.01f * val;

                            if (useMLPDrop)
                            {
                                float keepProb = 1.0f - network.MLPDropout;
                                float mask = dropoutRng.NextSingle() < keepProb ? (1.0f / keepProb) : 0f;
                                cache.MLPDropoutMask[qi, si, h, j] = mask;
                                activated *= mask;
                            }

                            cache.MLPHidden[qi, si, h, j] = activated;
                        }

                        float logit = network.B2[h];
                        for (int j = 0; j < hiddenDim; j++)
                            logit += network.W2[h, j] * cache.MLPHidden[qi, si, h, j];

                        cache.GateLogits[qi, si, h] = logit;
                        float gate = StableSigmoid(logit);
                        gate = network.ClampGate(gate);
                        cache.Gates[qi, si, h] = gate;

                        decayBias[qi, si, h] = -(baseRate * (1f - gate)) * normTd;
                    }
                }
            }

            return (decayBias, cache);
        }

        public float[,] ContentAwareCrossAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, float[,,] decayBias, out float[][,] attentionWeights, out float[][,] scoresPreSoftmax)
        {
            int queryLen = Q.GetLength(0);
            int keyLen = K.GetLength(0);
            int embDim = Q.GetLength(1);
            int headDim = embDim / numHeads;

            var output = new float[queryLen, embDim];
            attentionWeights = new float[numHeads][,];
            scoresPreSoftmax = new float[numHeads][,];

            for (int h = 0; h < numHeads; h++)
            {
                int offset = h * headDim;
                var scores = new float[queryLen, keyLen];
                var weights = new float[queryLen, keyLen];

                for (int q = 0; q < queryLen; q++)
                {
                    float max = float.NegativeInfinity;

                    for (int s = 0; s < keyLen; s++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                            dot += Q[q, offset + d] * K[s, offset + d];

                        float score = dot * scale;
                        if (decayBias != null)
                            score += decayBias[q, s, h];

                        scores[q, s] = score;
                        if (!float.IsNegativeInfinity(score) && score > max)
                            max = score;
                    }

                    if (float.IsNegativeInfinity(max))
                        continue;

                    float sum = 0f;
                    for (int s = 0; s < keyLen; s++)
                    {
                        if (float.IsNegativeInfinity(scores[q, s]))
                        {
                            weights[q, s] = 0f;
                            continue;
                        }

                        float w = MathF.Exp(scores[q, s] - max);
                        weights[q, s] = w;
                        sum += w;
                    }

                    if (sum > 0f)
                    {
                        float inv = 1f / sum;
                        for (int s = 0; s < keyLen; s++)
                            weights[q, s] *= inv;
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        float val = 0f;
                        for (int s = 0; s < keyLen; s++)
                            val += weights[q, s] * V[s, offset + d];
                        output[q, offset + d] = val;
                    }
                }

                attentionWeights[h] = weights;
                scoresPreSoftmax[h] = scores;
            }

            return output;
        }

        public float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V, float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TacamtBlock block, BlockCache bc, int PriceEmbeddingDim, int PriceNumHeads, bool enableDecayBias = true, bool isTraining = false, Random dropoutRng = null)
        {
            if (Q == null) throw new ArgumentNullException(nameof(Q));
            if (K == null) throw new ArgumentNullException(nameof(K));
            if (V == null) throw new ArgumentNullException(nameof(V));
            if (block == null) throw new ArgumentNullException(nameof(block));
            if (bc == null) throw new ArgumentNullException(nameof(bc));

            int queryLen = Q.GetLength(0);
            int keyLen = K.GetLength(0);
            int ed = PriceEmbeddingDim;
            int nh = PriceNumHeads;
            int hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);

            float[,,] decayBias = null;

            if (enableDecayBias && timeDiffs != null)
            {
                var (bias, decayCache) = ContentAwareDecayForward(
                    queryEmbeddings,
                    keyEmbeddings,
                    timeDiffs,
                    keyTimesFromRef,
                    block.DecayNetwork,
                    isTraining,
                    dropoutRng);

                decayBias = bias;
                bc.DecayCache = decayCache;
            }
            else
            {
                decayBias = null;
                bc.DecayCache = null;
            }

            var output = new float[queryLen, ed];
            var attentionWeights = new float[nh][,];
            var scoresPreSoftmax = new float[nh][,];

            for (int h = 0; h < nh; h++)
            {
                int offset = h * hd;
                var scores = new float[queryLen, keyLen];
                var weights = new float[queryLen, keyLen];

                for (int q = 0; q < queryLen; q++)
                {
                    float max = float.NegativeInfinity;

                    for (int s = 0; s < keyLen; s++)
                    {
                        // timeDiffs is a causal visibility mask even when Decay.Enabled=false.
                        bool valid = timeDiffs == null || timeDiffs[q, s] >= 0f;

                        if (!valid)
                        {
                            scores[q, s] = float.NegativeInfinity;
                            continue;
                        }

                        float dot = 0f;
                        for (int d = 0; d < hd; d++)
                            dot += Q[q, offset + d] * K[s, offset + d];

                        float score = dot * scale;
                        if (decayBias != null)
                            score += decayBias[q, s, h];

                        scores[q, s] = score;
                        if (!float.IsNegativeInfinity(score) && score > max)
                            max = score;
                    }

                    if (float.IsNegativeInfinity(max))
                        continue;

                    float sum = 0f;
                    for (int s = 0; s < keyLen; s++)
                    {
                        if (float.IsNegativeInfinity(scores[q, s]))
                        {
                            weights[q, s] = 0f;
                            continue;
                        }

                        float w = MathF.Exp(scores[q, s] - max);
                        weights[q, s] = w;
                        sum += w;
                    }

                    if (sum > 0f)
                    {
                        float inv = 1f / sum;
                        for (int s = 0; s < keyLen; s++)
                            weights[q, s] *= inv;
                    }

                    for (int d = 0; d < hd; d++)
                    {
                        float val = 0f;
                        for (int s = 0; s < keyLen; s++)
                            val += weights[q, s] * V[s, offset + d];

                        output[q, offset + d] = val;
                    }
                }

                attentionWeights[h] = weights;
                scoresPreSoftmax[h] = scores;
            }

            bc.CrossAttentionWeights = attentionWeights;
            bc.CrossScoresPreSoftmax = scoresPreSoftmax;
            return output;
        }

        public void Matrix3DScaleInPlace(float[,,] matrix, float scale)
        {
            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);

            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {

                        matrix[i, j, k] *= scale;
                    }
                }
            }
        }

        public float MatrixSquaredNorm3D(float[,,] matrix)
        {
            float sum = 0;
            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);
            for (int i = 0; i < d0; i++)
            {

                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        sum += matrix[i, j, k] * matrix[i, j, k];
                    }
                }
            }
            return sum;
        }



        #region MMTAC

        public float[] ProjectGlobalFeatures(float[] globalFeatures, float[,] projection, float[] bias)
        {
            if (globalFeatures == null)
            {
                throw new ArgumentNullException(nameof(globalFeatures));
            }

            if (projection == null)
            {
                throw new ArgumentNullException(nameof(projection));
            }
            if (bias == null)
            {
                throw new ArgumentNullException(nameof(bias));
            }

            int ed = projection.GetLength(0);
            int gd = projection.GetLength(1);

            if (globalFeatures.Length != gd)
            {
                throw new ArgumentException($"Expected global feature length {gd}, got {globalFeatures.Length}.");
            }
            if (bias.Length != ed)
            {
                throw new ArgumentException($"Expected bias length {ed}, got {bias.Length}.");
            }

            var output = new float[ed];

            for (int d = 0; d < ed; d++)
            {
                float sum = bias[d];

                for (int g = 0; g < gd; g++)
                {
                    sum += projection[d, g] * globalFeatures[g];
                }

                output[d] = sum;
            }

            return output;
        }

        public float[,] EmbedTokenIds(int[] tokenIds, float[,] embedding, int embeddingDim)
        {
            if (tokenIds == null || tokenIds.Length == 0)
            {
                return new float[0, embeddingDim];
            }
            if (embedding == null)
            {
                throw new ArgumentNullException(nameof(embedding));
            }

            int vocabSize = embedding.GetLength(0);

            if (embedding.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("Embedding dimension mismatch.", nameof(embedding));
            }

            var output = new float[tokenIds.Length, embeddingDim];

            for (int i = 0; i < tokenIds.Length; i++)
            {
                int tokenId = tokenIds[i];

                if ((uint)tokenId >= (uint)vocabSize)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token out of range: {tokenId}");
                }

                for (int d = 0; d < embeddingDim; d++)
                {
                    output[i, d] = embedding[tokenId, d];
                }
            }

            return output;
        }

        public float[] MeanPoolRows(float[,] matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var output = new float[cols];

            if (rows == 0)
            {
                return output;
            }

            float inv = 1.0f / rows;

            for (int d = 0; d < cols; d++)
            {
                float sum = 0f;

                for (int r = 0; r < rows; r++)
                {
                    sum += matrix[r, d];
                }

                output[d] = sum * inv;
            }

            return output;
        }

        public (float[,] contextHidden, float[] contextTimes, int numGlobal, int numNews) BuildMmtacContext(float[,] newsHidden, float[] newsTimes, float[] globalToken, float[,] contextTypeEmbedding)
        {
            if (contextTypeEmbedding == null)
            {
                throw new ArgumentNullException(nameof(contextTypeEmbedding));
            }

            int ed = contextTypeEmbedding.GetLength(1);
            int numGlobal = globalToken != null ? 1 : 0;
            int numNews = newsHidden != null ? newsHidden.GetLength(0) : 0;
            int total = numGlobal + numNews;

            if (total == 0)
            {
                return (null, null, 0, 0);
            }

            if (globalToken != null && globalToken.Length != ed)
            {
                throw new ArgumentException("globalToken length must match embedding dimension.", nameof(globalToken));
            }

            if (newsHidden != null && newsHidden.GetLength(1) != ed)
            {
                throw new ArgumentException("newsHidden embedding dimension mismatch.", nameof(newsHidden));
            }

            if (newsTimes != null && newsTimes.Length != numNews)
            {
                throw new ArgumentException("newsTimes length must match newsHidden row count.", nameof(newsTimes));
            }

            var contextHidden = new float[total, ed];
            var contextTimes = new float[total];

            int row = 0;

            if (globalToken != null)
            {
                for (int d = 0; d < ed; d++)
                {
                    contextHidden[row, d] = globalToken[d] + contextTypeEmbedding[2, d];
                }

                contextTimes[row] = 0f;
                row++;
            }

            for (int i = 0; i < numNews; i++)
            {
                for (int d = 0; d < ed; d++)
                {
                    contextHidden[row, d] = newsHidden[i, d] + contextTypeEmbedding[0, d];
                }

                contextTimes[row] = newsTimes != null ? newsTimes[i] : 0f;
                row++;
            }

            return (contextHidden, contextTimes, numGlobal, numNews);
        }

        public (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence, float[,] regressionLogits, float[] rangeLogits, float[] qualityLogits) ProjectMmtacOutputHeads(float[,] hidden, float[,] regressionProjection, float[] regressionBias, float[,] rangeProjection, float[] rangeBias, float[,] qualityProjection, float[] qualityBias, float[,] directionProjection, float[] directionBias, float[,] midDirectionProjection, float[] midDirectionBias, float[,] confidenceProjection, float[] confidenceBias, bool useConfidenceHead)
        {
            if (hidden == null)
            {
                throw new ArgumentNullException(nameof(hidden));
            }

            int sl = hidden.GetLength(0);
            int ed = hidden.GetLength(1);
            const int rDim = 3;

            var rawRegression = ProjectOutputBatch(hidden, regressionProjection, regressionBias, sl, rDim);

            var regression = new float[sl, rDim];
            var range = new float[sl, 1];
            var rangeLogits = new float[sl];

            for (int t = 0; t < sl; t++)
            {
                float upBase = Softplus(rawRegression[t, 0]);
                float downBase = Softplus(rawRegression[t, 1]);

                float l = rangeBias[0];

                for (int k = 0; k < ed; k++)
                    l += rangeProjection[0, k] * hidden[t, k];

                rangeLogits[t] = l;

                float rangeValue = Softplus(l);
                float den = upBase + downBase;
                float upShare = den > 1e-6f ? upBase / den : 0.5f;
                float downShare = 1f - upShare;
                float close = rawRegression[t, 2];

                regression[t, 0] = close + rangeValue * upShare;
                regression[t, 1] = close - rangeValue * downShare;
                regression[t, 2] = close;
                range[t, 0] = rangeValue;
            }

            var quality = new float[sl, 1];
            var qualityLogits = new float[sl];

            for (int t = 0; t < sl; t++)
            {
                float l = qualityBias[0];

                for (int k = 0; k < ed; k++)
                    l += qualityProjection[0, k] * hidden[t, k];

                qualityLogits[t] = l;
                quality[t, 0] = StableSigmoid(l);
            }

            var direction = new float[sl, 1];

            for (int t = 0; t < sl; t++)
            {
                float l = directionBias[0];

                for (int k = 0; k < ed; k++)
                    l += directionProjection[0, k] * hidden[t, k];

                direction[t, 0] = StableSigmoid(l);
            }

            var midDirection = new float[sl, 1];

            for (int t = 0; t < sl; t++)
            {
                float l = midDirectionBias[0];

                for (int k = 0; k < ed; k++)
                    l += midDirectionProjection[0, k] * hidden[t, k];

                midDirection[t, 0] = StableSigmoid(l);
            }

            float[,] confidence = null;

            if (useConfidenceHead)
            {
                confidence = new float[sl, 1];

                for (int t = 0; t < sl; t++)
                {
                    float l = confidenceBias[0];

                    for (int k = 0; k < ed; k++)
                        l += confidenceProjection[0, k] * hidden[t, k];

                    confidence[t, 0] = StableSigmoid(l);
                }
            }

            return (regression, range, quality, direction, midDirection, confidence, rawRegression, rangeLogits, qualityLogits);
        }

        public float[] SoftmaxVector(float[] scores)
        {
            if (scores == null)
            {
                throw new ArgumentNullException(nameof(scores));
            }

            var output = new float[scores.Length];

            if (scores.Length == 0)
            {
                return output;
            }

            float max = float.NegativeInfinity;

            for (int i = 0; i < scores.Length; i++)
            {
                if (scores[i] > max)
                {
                    max = scores[i];
                }
            }

            float sum = 0f;

            for (int i = 0; i < scores.Length; i++)
            {
                output[i] = MathF.Exp(scores[i] - max);
                sum += output[i];
            }

            if (sum <= 0f)
            {
                return output;
            }

            float inv = 1f / sum;

            for (int i = 0; i < output.Length; i++)
            {
                output[i] *= inv;
            }

            return output;
        }

        public (float[,] dQ, float[,] dK, float[,] dV, float[,,] dDecayBias) BackpropTimeDecayedAttention(float[,] q, float[,] k, float[,] v, float[,] dOutput, float[][,] attentionWeights, float[,] timeDiffs, int embeddingDim, int numHeads)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (k == null) throw new ArgumentNullException(nameof(k));
            if (v == null) throw new ArgumentNullException(nameof(v));
            if (dOutput == null) throw new ArgumentNullException(nameof(dOutput));
            if (attentionWeights == null) throw new ArgumentNullException(nameof(attentionWeights));

            int queryLen = q.GetLength(0);
            int keyLen = k.GetLength(0);
            int headDim = embeddingDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);

            var dQ = new float[queryLen, embeddingDim];
            var dK = new float[keyLen, embeddingDim];
            var dV = new float[keyLen, embeddingDim];

            float[,,] dDecayBias = timeDiffs != null
                ? new float[queryLen, keyLen, numHeads]
                : null;

            for (int h = 0; h < numHeads; h++)
            {
                int offset = h * headDim;
                var weights = attentionWeights[h];

                for (int p = 0; p < queryLen; p++)
                {
                    var dWeights = new float[keyLen];

                    for (int s = 0; s < keyLen; s++)
                    {
                        float dot = 0f;

                        for (int d = 0; d < headDim; d++)
                        {
                            int di = offset + d;

                            dV[s, di] += weights[p, s] * dOutput[p, di];
                            dot += dOutput[p, di] * v[s, di];
                        }

                        dWeights[s] = dot;
                    }

                    float dotWeightGrad = 0f;

                    for (int s = 0; s < keyLen; s++)
                        dotWeightGrad += weights[p, s] * dWeights[s];

                    for (int s = 0; s < keyLen; s++)
                    {
                        float dScore = weights[p, s] * (dWeights[s] - dotWeightGrad);

                        for (int d = 0; d < headDim; d++)
                        {
                            int di = offset + d;

                            dQ[p, di] += dScore * scale * k[s, di];
                            dK[s, di] += dScore * scale * q[p, di];
                        }

                        if (dDecayBias != null)
                            dDecayBias[p, s, h] += dScore;
                    }
                }
            }

            return (dQ, dK, dV, dDecayBias);
        }

        #endregion

        #endregion

        #region Tokenizer Acceleration

        public string[] PreTokenize(string text)
        {
            var tokens = new List<string>();
            var currentToken = new StringBuilder();

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];

                if (char.IsLetterOrDigit(c))
                {
                    currentToken.Append(c);
                }
                else
                {
                    if (currentToken.Length > 0)
                    {
                        tokens.Add(currentToken.ToString());
                        currentToken.Clear();
                    }

                    if (!char.IsWhiteSpace(c))
                    {
                        tokens.Add(c.ToString());
                    }
                }
            }

            if (currentToken.Length > 0)
            {
                tokens.Add(currentToken.ToString());
            }

            return tokens.ToArray();
        }

        private static string ToCharacterSequence(string word)
        {
            if (string.IsNullOrEmpty(word))
            {
                return string.Empty;
            }

            if (word.Length == 1)
            {
                return word;
            }

            var sb = new StringBuilder(word.Length * 2 - 1);
            sb.Append(word[0]);

            for (int i = 1; i < word.Length; i++)
            {
                sb.Append(' ');
                sb.Append(word[i]);
            }

            return sb.ToString();
        }

        public Dictionary<string, int> GetWordFrequencies(string[] texts, bool lowerCase)
        {
            var wordFreqs = new Dictionary<string, int>();

            foreach (var text in texts)
            {
                var processedText = lowerCase ? text.ToLowerInvariant() : text;

                var words = PreTokenize(processedText);

                foreach (var word in words)
                {
                    if (string.IsNullOrWhiteSpace(word))
                    {
                        continue;
                    }
                    string charSeq = ToCharacterSequence(word);

                    if (wordFreqs.TryGetValue(charSeq, out int count))
                    {
                        wordFreqs[charSeq] = count + 1;
                    }
                    else
                    {
                        wordFreqs[charSeq] = 1;
                    }
                }
            }

            return wordFreqs;
        }

        public HashSet<string> BuildCharacterVocabulary(Dictionary<string, int> wordFreqs)
        {
            var chars = new HashSet<string>();

            foreach (var word in wordFreqs.Keys)
            {
                foreach (var ch in word.Split(' '))
                {
                    if (!string.IsNullOrEmpty(ch))
                    {
                        chars.Add(ch);
                    }
                }
            }

            return chars;
        }

        public List<string> ApplyMerge(List<string> word, string left, string right)
        {
            var result = new List<string>();
            int i = 0;

            while (i < word.Count)
            {
                if (i < word.Count - 1 && word[i] == left && word[i + 1] == right)
                {
                    result.Add(left + right);
                    i += 2;
                }
                else
                {
                    result.Add(word[i]);
                    i++;
                }
            }

            return result;
        }

        public List<int> EncodeWord(string word, Dictionary<(string, string), int> mergePriority, Dictionary<string, int> vocabToId, int unkTokenId)
        {
            var parts = word.Select(c => c.ToString()).ToList();

            while (parts.Count > 1)
            {
                (string, string)? bestMerge = null;
                int bestPriority = int.MaxValue;
                int bestPos = -1;

                for (int i = 0; i < parts.Count - 1; i++)
                {
                    var pair = (parts[i], parts[i + 1]);

                    if (mergePriority.TryGetValue(pair, out int priority))
                    {
                        if (priority < bestPriority)
                        {
                            bestMerge = pair;
                            bestPriority = priority;
                            bestPos = i;
                        }
                    }
                }

                if (bestMerge == null)
                {
                    break;
                }

                var (left, right) = bestMerge.Value;
                parts[bestPos] = left + right;
                parts.RemoveAt(bestPos + 1);
            }

            var tokenIds = new List<int>();

            foreach (var part in parts)
            {
                if (vocabToId.TryGetValue(part, out int id))
                {
                    tokenIds.Add(id);
                }
                else
                {
                    foreach (var ch in part)
                    {
                        var chStr = ch.ToString();

                        if (vocabToId.TryGetValue(chStr, out int chId))
                        {
                            tokenIds.Add(chId);
                        }
                        else
                        {
                            tokenIds.Add(unkTokenId);
                        }
                    }
                }
            }

            return tokenIds;
        }

        public Dictionary<(string left, string right), int> CountPairFrequencies(Dictionary<List<string>, int> words)
        {
            var pairCounts = new Dictionary<(string left, string right), int>();

            foreach (var kv in words)
            {
                var word = kv.Key;
                int freq = kv.Value;

                for (int i = 0; i < word.Count - 1; i++)
                {
                    var pair = (word[i], word[i + 1]);

                    if (pairCounts.ContainsKey(pair))
                    {
                        pairCounts[pair] += freq;
                    }
                    else
                    {
                        pairCounts[pair] = freq;
                    }
                }
            }

            return pairCounts;
        }

        public ((string left, string right) pair, int frequency) SelectBestPair(Dictionary<(string left, string right), int> pairCounts, int minFrequency)
        {
            if (pairCounts == null || pairCounts.Count == 0)
            {
                return ((null, null), 0);
            }

            var bestPair = pairCounts
                .Where(kv => kv.Value >= minFrequency)
                .OrderByDescending(kv => kv.Value)
                .ThenBy(kv => kv.Key.left + kv.Key.right)
                .FirstOrDefault();

            return (bestPair.Key, bestPair.Value);
        }

        public Dictionary<List<string>, int> ApplyMergeToVocabulary(Dictionary<List<string>, int> words, string left, string right)
        {
            var newWords = new Dictionary<List<string>, int>(new ListEqualityComparer<string>());

            foreach (var kv in words)
            {
                var newWord = ApplyMerge(kv.Key, left, right);

                if (newWords.ContainsKey(newWord))
                {
                    newWords[newWord] += kv.Value;
                }
                else
                {
                    newWords[newWord] = kv.Value;
                }
            }

            return newWords;
        }

        public string DecodeTokens(int[] tokenIds, Dictionary<int, string> idToVocab, string unkToken, bool skipSpecialTokens)
        {
            if (tokenIds == null || tokenIds.Length == 0)
            {
                return string.Empty;
            }

            var sb = new StringBuilder();

            foreach (var id in tokenIds)
            {
                if (!idToVocab.TryGetValue(id, out var token))
                {
                    token = unkToken;
                }
                if (skipSpecialTokens && token.StartsWith("<|") && token.EndsWith("|>"))
                {
                    continue;
                }

                sb.Append(token);
            }

            return sb.ToString();
        }

        public int[] PadOrTruncate(int[] tokenIds, int maxLength, bool addSpecialTokens, int padTokenId, int endTokenId)
        {
            if (tokenIds.Length > maxLength)
            {
                var truncated = new int[maxLength];
                Array.Copy(tokenIds, truncated, maxLength);

                if (addSpecialTokens)
                {
                    truncated[maxLength - 1] = endTokenId;
                }

                return truncated;
            }

            if (tokenIds.Length < maxLength)
            {
                var padded = new int[maxLength];
                Array.Copy(tokenIds, padded, tokenIds.Length);

                for (int i = tokenIds.Length; i < maxLength; i++)
                {
                    padded[i] = padTokenId;
                }

                return padded;
            }

            return tokenIds;
        }


        #endregion

        #region Rotary Position Embeddings
        public void ApplyRotaryPositionEmbeddingInPlace(float[,] matrix, int numHeads, float baseTheta, bool inverse)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            if (numHeads <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numHeads));
            }

            if (baseTheta <= 0f || float.IsNaN(baseTheta) || float.IsInfinity(baseTheta))
            {
                throw new ArgumentOutOfRangeException(nameof(baseTheta));
            }

            int embeddingDim = matrix.GetLength(1);

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by number of heads.");
            }

            int headDim = embeddingDim / numHeads;

            if ((headDim & 1) != 0)
            {
                throw new ArgumentException("RoPE requires an even per-head dimension.");
            }

            for (int head = 0; head < numHeads; head++)
            {
                int startCol = head * headDim;

                ApplyRotaryPositionEmbeddingHeadCoreInPlace(
                    matrix,
                    startCol,
                    headDim,
                    baseTheta,
                    inverse);
            }
        }
        public void ApplyRotaryPositionEmbeddingHeadInPlace(float[,] matrix, int startCol, int headDim, float baseTheta, bool inverse)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            if (startCol < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(startCol));
            }

            if (headDim < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(headDim));
            }

            if ((headDim & 1) != 0)
            {
                throw new ArgumentException("RoPE requires an even per-head dimension.", nameof(headDim));
            }

            if (baseTheta <= 0f || float.IsNaN(baseTheta) || float.IsInfinity(baseTheta))
            {
                throw new ArgumentOutOfRangeException(nameof(baseTheta));
            }

            int embeddingDim = matrix.GetLength(1);

            if (startCol + headDim > embeddingDim)
            {
                throw new ArgumentException("Head range exceeds matrix embedding dimension.");
            }

            ApplyRotaryPositionEmbeddingHeadCoreInPlace(
                matrix,
                startCol,
                headDim,
                baseTheta,
                inverse);
        }
        private static void ApplyRotaryPositionEmbeddingHeadCoreInPlace(float[,] matrix, int startCol, int headDim, float baseTheta, bool inverse)
        {
            int seqLen = matrix.GetLength(0);
            int pairCount = headDim / 2;

            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int pair = 0; pair < pairCount; pair++)
                {
                    int evenCol = startCol + (pair * 2);
                    int oddCol = evenCol + 1;

                    float theta = pos / MathF.Pow(baseTheta, (2f * pair) / headDim);
                    float cos = MathF.Cos(theta);
                    float sin = MathF.Sin(theta);

                    if (inverse)
                    {
                        sin = -sin;
                    }

                    float x0 = matrix[pos, evenCol];
                    float x1 = matrix[pos, oddCol];

                    matrix[pos, evenCol] = (x0 * cos) - (x1 * sin);
                    matrix[pos, oddCol] = (x0 * sin) + (x1 * cos);
                }
            }
        }
        #endregion
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
        private float StableSigmoid(float x)
        {
            if (x >= 0)
            {
                float ex = MathF.Exp(-x);
                return 1f / (1f + ex);
            }
            else
            {
                float ex = MathF.Exp(x);
                return ex / (1f + ex);
            }
        }
        private float Softplus(float x)
        {
            if (x > 20f)
            {
                return x;
            }
            if (x < -20f)
            {
                return MathF.Exp(x);
            }
            return MathF.Log(1f + MathF.Exp(x));
        }
        public void Dispose() { }
    }
}