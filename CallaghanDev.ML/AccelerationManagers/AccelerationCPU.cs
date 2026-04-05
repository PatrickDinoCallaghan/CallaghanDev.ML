using CallaghanDev.ML.Enums;
using static CallaghanDev.ML.Functions;
using CallaghanDev.ML.Transformers.TACAMT;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationCPU : IAccelerationManager
    {
        public AccelerationCPU()
        {
        }

        // I realize this behaves like a transpose rather than a plain dot product-but that’s exactly what we want, and it’s correct.
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
                if (sum > 0f)
                {
                    for (int j = 0; j < cols; j++)
                    {

                        result[i, j] = result[i, j] / sum;
                    }
                }
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
            int seqLen = inputMatrix.GetLength(0);

            // Delegate to offset-aware version
            return BatchDotProduct(weights, inputMatrix, rowStart: 0, rowCount: seqLen);
        }
        public float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix, int rowStart, int rowCount)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (inputMatrix == null) throw new ArgumentNullException(nameof(inputMatrix));

            if (rowStart < 0 || rowCount < 0)
                throw new ArgumentOutOfRangeException();

            if (rowStart + rowCount > inputMatrix.GetLength(0))
                throw new ArgumentException("Invalid row slice.");

            int outputDim = weights.GetLength(0);
            int inputDim = weights.GetLength(1);

            if (inputMatrix.GetLength(1) != inputDim)
                throw new ArgumentException(
                    $"Expected input columns {inputDim}, got {inputMatrix.GetLength(1)}");

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
        #region Multi-head attention

        public float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
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
                        attnWeights[i, j] = attnWeights[i, j]/ expSum;
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
                        //These bugs were symetric would never have seen them
                        //weightGrad[k, j] += input[i, k] * dOutVal;
                        weightGrad[j, k] += dOutVal * input[i, k];
                    }

                    biasGrad[j] += dOutVal;
                }

                for (int k = 0; k < embeddingDim; k++)
                {
                    float sum = 0;
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        //These bugs were symetric would never have seen them
                        //sum += dOutput[i, j] * weights[k, j];
                        sum += dOutput[i, j] * weights[j, k];
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
        public float[,] ContentAwareCrossAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, float[,,] decayBias, out float[][,] attentionWeights, out float[][,] scoresPreSoftmax)
        {
            int psl = Q.GetLength(0);
            int tsl = K.GetLength(0);
            int embDim = Q.GetLength(1);
            int headDim = embDim / numHeads;

            var output = new float[psl, embDim];
            attentionWeights = new float[numHeads][,];
            scoresPreSoftmax = new float[numHeads][,];

            for (int h = 0; h < numHeads; h++)
            {
                int si = h * headDim;
                var scores = new float[psl, tsl];
                var weights = new float[psl, tsl];

                for (int p = 0; p < psl; p++)
                {
                    //float mx = float.MinValue;
                    float mx = float.NegativeInfinity;


                    for (int s = 0; s < tsl; s++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                            dot += Q[p, si + d] * K[s, si + d];

                        float sc = dot * scale;
                        if (decayBias != null) sc += decayBias[p, s, h];
                        scores[p, s] = sc;
                        if (sc > mx) mx = sc;
                    }

                    float se = 0f;
                    for (int s = 0; s < tsl; s++)
                    {
                        weights[p, s] = MathF.Exp(scores[p, s] - mx);
                        se += weights[p, s];
                    }

                    if (se > 0f)
                        for (int s = 0; s < tsl; s++)
                            weights[p, s] /= se;

                    for (int d = 0; d < headDim; d++)
                    {
                        float v = 0f;
                        for (int s = 0; s < tsl; s++)
                            v += weights[p, s] * V[s, si + d];
                        output[p, si + d] = v;
                    }
                }

                attentionWeights[h] = weights;
                scoresPreSoftmax[h] = scores;
            }

            return output;
        }

        public float[,] FFNForwardBatch(
            float[,] input,
            int seqLen,
            int outputDim,
            Func<float[], float[]> forwardPassFn)
        {
            var result = new float[seqLen, outputDim];

            for (int i = 0; i < seqLen; i++)
            {
                var row = new float[input.GetLength(1)];
                for (int j = 0; j < input.GetLength(1); j++)
                    row[j] = input[i, j];

                var out_row = forwardPassFn(row);

                for (int j = 0; j < outputDim; j++)
                    result[i, j] = out_row[j];
            }

            return result;
        }

        public void ApplyContextTypeEmbedding(
            float[,] contextHidden,
            float[,] typeEmbedding,
            int[] typeIndices)
        {
            int n = contextHidden.GetLength(0);
            int embDim = contextHidden.GetLength(1);

            for (int i = 0; i < n; i++)
            {
                int t = typeIndices[i];
                for (int d = 0; d < embDim; d++)
                    contextHidden[i, d] += typeEmbedding[t, d];
            }
        }

        public float[,] ComputeTimeDiffMatrix(
            int priceSeqLen,
            float[] keyArrivalTimes)
        {
            int numKeys = keyArrivalTimes.Length;
            var td = new float[priceSeqLen, numKeys];

            for (int p = 0; p < priceSeqLen; p++)
                for (int s = 0; s < numKeys; s++)
                    td[p, s] = MathF.Abs(p - keyArrivalTimes[s]);

            return td;
        }

        public float[,] MeanPoolRows(
            float[,] hidden,
            int[] storyOffsets,
            int[] storyCounts,
            int numStories,
            int embeddingDim)
        {
            var result = new float[numStories, embeddingDim];

            for (int s = 0; s < numStories; s++)
            {
                int start = storyOffsets[s];
                int count = storyCounts[s];

                if (count <= 0) continue;

                float inv = 1f / count;

                for (int d = 0; d < embeddingDim; d++)
                {
                    float sum = 0f;
                    for (int t = start; t < start + count; t++)
                        sum += hidden[t, d];
                    result[s, d] = sum * inv;
                }
            }

            return result;
        }

        public float[,] EmbedWithBiasAndPositional(
            float[,] projected,
            float[] bias,
            float[,] positionalEncoding,
            int seqLen,
            int embeddingDim)
        {
            var result = new float[seqLen, embeddingDim];

            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embeddingDim; j++)
                    result[i, j] = projected[i, j] + bias[j] + positionalEncoding[i, j];

            return result;
        }

        public float[] ComputeMemoryAttentionScores(
            float[,] priceHidden,
            int lastPos,
            float[,] contextHidden,
            int totalCtx,
            float scale)
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

        public float[,] ProjectOutputBatch(
            float[,] hidden,
            float[,] outputProjection,
            float[] outputBias,
            int seqLen,
            int outputDim)
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
        public (float[,,] decayBias, ContentAwareDecayCache cache) ContentAwareDecayForward(
    float[,] queryEmbeddings,
    float[,] keyEmbeddings,
    float[,] timeDiffs,
    float[] keyTimesFromRef,
    ContentAwareDecayNetwork network,
    bool isTraining = false, Random dropoutRng = null)
        {
            int queryLen = timeDiffs.GetLength(0);
            int keyLen = timeDiffs.GetLength(1);
            int numHeads = network.NumHeads;
            int projDim = network.ProjectionDim;
            int contentDim = network.ContentDim;
            int hiddenDim = network.HiddenDim;
            int mlpInputDim = network.MLPInputDim;
            int numBases = network.NumTimeBases;
            int rawDim = network.TimeRawDim;

            // Allocate cache
            var cache = new ContentAwareDecayCache
            {
                QueryEmbeddings = queryEmbeddings,
                KeyEmbeddings = keyEmbeddings,
                TimeDiffs = timeDiffs,
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

            bool useMemAttnDrop = isTraining && network.MemoryAttentionDropout > 0 && dropoutRng != null;
            bool useMLPDrop = isTraining && network.MLPDropout > 0 && dropoutRng != null;

            if (useMemAttnDrop)
                cache.MemAttnDropoutMask = new float[numHeads, keyLen, keyLen];
            if (useMLPDrop)
                cache.MLPDropoutMask = new float[queryLen, keyLen, numHeads, hiddenDim];

            var decayBias = new float[queryLen, keyLen, numHeads];

            // Process each head sequentially
            for (int h = 0; h < numHeads; h++)
            {
                // 1. Project queries
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

                // 2. Project keys
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

                // 3. Multi-scale sinusoidal time encoding
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

                // 4. Memory interaction attention
                float memScale = 1.0f / MathF.Sqrt(projDim);

                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        cache.MemAttnQInput[h, s, p] = cache.KeyProj[h, s, p] + cache.TimeEncoding[h, s, p];
                        cache.MemAttnKInput[h, s, p] = cache.KeyProj[h, s, p] + cache.TimeEncoding[h, s, p];
                    }
                }

                for (int i = 0; i < keyLen; i++)
                {
                    float maxScore = float.MinValue;
                    var scores = new float[keyLen];

                    for (int j = 0; j < keyLen; j++)
                    {
                        float dot = 0;
                        for (int p = 0; p < projDim; p++)
                            dot += cache.MemAttnQInput[h, i, p] * cache.MemAttnKInput[h, j, p];
                        scores[j] = dot * memScale;
                        if (scores[j] > maxScore) maxScore = scores[j];
                    }

                    float sumExp = 0;
                    for (int j = 0; j < keyLen; j++)
                    {
                        cache.MemAttnWeights[h, i, j] = MathF.Exp(scores[j] - maxScore);
                        sumExp += cache.MemAttnWeights[h, i, j];
                    }
                    if (sumExp > 0)
                    {
                        for (int j = 0; j < keyLen; j++)
                            cache.MemAttnWeights[h, i, j] /= sumExp;
                    }

                    // Dropout on memory attention
                    if (useMemAttnDrop)
                    {
                        float keepProb = 1.0f - network.MemoryAttentionDropout;
                        float dscale = 1.0f / keepProb;
                        for (int j = 0; j < keyLen; j++)
                        {
                            float mask = dropoutRng.NextSingle() < keepProb ? dscale : 0f;
                            cache.MemAttnDropoutMask[h, i, j] = mask;
                            cache.MemAttnWeights[h, i, j] *= mask;
                        }
                    }

                    // Weighted sum
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = 0;
                        for (int j = 0; j < keyLen; j++)
                            val += cache.MemAttnWeights[h, i, j] * cache.KeyProj[h, j, p];
                        cache.MemAttnOutput[h, i, p] = val;
                    }
                }

                // 5. Output projection (refined key)
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

                // 6. Gating MLP
                for (int qi = 0; qi < queryLen; qi++)
                {
                    for (int si = 0; si < keyLen; si++)
                    {
                        float td = timeDiffs[qi, si];
                        float logTd = MathF.Log(1f + td);

                        // Build MLP input
                        int idx = 0;
                        for (int p = 0; p < projDim; p++) cache.MLPInput[qi, si, h, idx++] = cache.QueryProj[h, qi, p];
                        for (int p = 0; p < projDim; p++) cache.MLPInput[qi, si, h, idx++] = cache.RefinedKey[h, si, p];
                        for (int p = 0; p < projDim; p++) cache.MLPInput[qi, si, h, idx++] = cache.QueryProj[h, qi, p] * cache.RefinedKey[h, si, p];
                        cache.MLPInput[qi, si, h, idx++] = td;
                        cache.MLPInput[qi, si, h, idx++] = logTd;

                        // Hidden layer (ReLU)
                        for (int j = 0; j < hiddenDim; j++)
                        {
                            float val = network.B1[h, j];
                            for (int k = 0; k < mlpInputDim; k++)
                                val += network.W1[h, j, k] * cache.MLPInput[qi, si, h, k];
                            cache.MLPHiddenPreAct[qi, si, h, j] = val;
                            float activated = val > 0 ? val : 0;

                            // Dropout on MLP hidden
                            if (useMLPDrop)
                            {
                                float keepProb = 1.0f - network.MLPDropout;
                                float mask = dropoutRng.NextSingle() < keepProb ? (1.0f / keepProb) : 0f;
                                cache.MLPDropoutMask[qi, si, h, j] = mask;
                                activated *= mask;
                            }
                            cache.MLPHidden[qi, si, h, j] = activated;
                        }

                        // Output layer
                        float logit = network.B2[h];
                        for (int j = 0; j < hiddenDim; j++)
                            logit += network.W2[h, j] * cache.MLPHidden[qi, si, h, j];
                        cache.GateLogits[qi, si, h] = logit;

                        float gate = StableSigmoid(logit);
                        cache.Gates[qi, si, h] = gate;

                        float baseRate = MathF.Exp(network.LogBaseDecayRate[h]);
                        decayBias[qi, si, h] = -(baseRate * gate) * td;
                    }
                }
            }

            return (decayBias, cache);
        }

        private static float StableSigmoid(float x)
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

        public float[,] ContentAwareCrossAttentionWithCache(
            float[,] Q,
            float[,] K,
            float[,] V,
            float[,] timeDiffs,
            float[] keyTimesFromRef,
            float[,] queryEmbeddings,
            float[,] keyEmbeddings,
    CallaghanDev.ML.Transformers.TACAMT.TransformerBlock block,
            BlockCache bc,
            int PriceEmbeddingDim,
            int PriceNumHeads,
            bool isTraining = false,
            Random dropoutRng = null)
        {
            int psl = Q.GetLength(0);
            int tsl = K.GetLength(0);
            int ed = PriceEmbeddingDim;
            int nh = PriceNumHeads;
            int hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);

            float[,,] decayBias = null;

            if (timeDiffs != null)
            {
                var (bias, decayCache) = ContentAwareDecayForward(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, block.DecayNetwork, isTraining, dropoutRng);
                decayBias = bias;
                bc.DecayCache = decayCache;
            }

            float[][,] attentionWeights;
            float[][,] scoresPreSoftmax;

            var output = ContentAwareCrossAttentionForward(Q, K, V, nh, scale, decayBias,out attentionWeights, out scoresPreSoftmax);

            bc.CrossAttentionWeights = attentionWeights;
            bc.CrossScoresPreSoftmax = scoresPreSoftmax;

            return output;
        }


        public void VectorAccumulate(float[] target, float[] source) => AccumulateVectorGradients(target, source);

        public void MatrixAddInPlace(float[,] target, float[,] addend) => MatrixAccumulate(target, addend);

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
    }

}