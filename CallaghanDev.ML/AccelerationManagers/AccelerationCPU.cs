using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.TACAMT;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using static CallaghanDev.ML.Functions;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationCPU : IAccelerationManager
    {
        public AccelerationCPU()
        {
        }


        #region CPU hot-path helpers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Span<float> FlatSpan(float[,] matrix)
        {
            return matrix.Length == 0 ? Span<float>.Empty : MemoryMarshal.CreateSpan(ref matrix[0, 0], matrix.Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ReadOnlySpan<float> FlatReadOnlySpan(float[,] matrix)
        {
            return matrix.Length == 0 ? ReadOnlySpan<float>.Empty : MemoryMarshal.CreateSpan(ref matrix[0, 0], matrix.Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Span<float> FlatSpan(float[,,] matrix)
        {
            return matrix.Length == 0 ? Span<float>.Empty : MemoryMarshal.CreateSpan(ref matrix[0, 0, 0], matrix.Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ReadOnlySpan<float> FlatReadOnlySpan(float[,,] matrix)
        {
            return matrix.Length == 0 ? ReadOnlySpan<float>.Empty : MemoryMarshal.CreateSpan(ref matrix[0, 0, 0], matrix.Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ReadOnlySpan<bool> FlatReadOnlySpan(bool[,] matrix)
        {
            return matrix.Length == 0 ? ReadOnlySpan<bool>.Empty : MemoryMarshal.CreateSpan(ref matrix[0, 0], matrix.Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Offset2D(int row, int cols)
        {
            return row * cols;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Offset3D(int i, int j, int dim1, int dim2)
        {
            return ((i * dim1) + j) * dim2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Dot(ReadOnlySpan<float> left, int leftOffset, ReadOnlySpan<float> right, int rightOffset, int length)
        {
            float sum = 0.0f;
            int i = 0;
            int end4 = length & ~3;

            for (; i < end4; i += 4)
            {
                sum += left[leftOffset + i] * right[rightOffset + i];
                sum += left[leftOffset + i + 1] * right[rightOffset + i + 1];
                sum += left[leftOffset + i + 2] * right[rightOffset + i + 2];
                sum += left[leftOffset + i + 3] * right[rightOffset + i + 3];
            }

            for (; i < length; i++)
            {
                sum += left[leftOffset + i] * right[rightOffset + i];
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AddScaledRow(Span<float> destination, int destOffset, ReadOnlySpan<float> source, int sourceOffset, int length, float scale)
        {
            int i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                int vectorWidth = Vector<float>.Count;
                int endVector = length - (length % vectorWidth);
                if (endVector > 0)
                {
                    var scaleVector = new Vector<float>(scale);
                    for (; i < endVector; i += vectorWidth)
                    {
                        var dstVector = new Vector<float>(destination.Slice(destOffset + i, vectorWidth));
                        var srcVector = new Vector<float>(source.Slice(sourceOffset + i, vectorWidth));
                        (dstVector + (srcVector * scaleVector)).CopyTo(destination.Slice(destOffset + i, vectorWidth));
                    }
                }
            }

            int end4 = length - ((length - i) & 3);
            for (; i < end4; i += 4)
            {
                destination[destOffset + i] += source[sourceOffset + i] * scale;
                destination[destOffset + i + 1] += source[sourceOffset + i + 1] * scale;
                destination[destOffset + i + 2] += source[sourceOffset + i + 2] * scale;
                destination[destOffset + i + 3] += source[sourceOffset + i + 3] * scale;
            }

            for (; i < length; i++)
            {
                destination[destOffset + i] += source[sourceOffset + i] * scale;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float StableSigmoidCore(float x)
        {
            if (x >= 0f)
            {
                float ex = MathF.Exp(-x);
                return 1f / (1f + ex);
            }

            float exNeg = MathF.Exp(x);
            return exNeg / (1f + exNeg);
        }

        #endregion

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
            if (rowsA == 0 || colsB == 0 || colsA == 0)
            {
                return C;
            }

            ReadOnlySpan<float> a = FlatReadOnlySpan(A);
            ReadOnlySpan<float> b = FlatReadOnlySpan(B);
            Span<float> c = FlatSpan(C);

            int transposedLength = colsB * colsA;
            float[] rented = ArrayPool<float>.Shared.Rent(transposedLength);

            try
            {
                Span<float> bT = rented.AsSpan(0, transposedLength);

                for (int k = 0; k < colsA; k++)
                {
                    int bRow = k * colsB;
                    for (int j = 0; j < colsB; j++)
                    {
                        bT[(j * colsA) + k] = b[bRow + j];
                    }
                }

                for (int i = 0; i < rowsA; i++)
                {
                    int aRow = i * colsA;
                    int cRow = i * colsB;

                    for (int j = 0; j < colsB; j++)
                    {
                        c[cRow + j] = Dot(a, aRow, bT, j * colsA, colsA);
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
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
            if (rowsA == 0 || rowsB == 0 || colsA == 0)
            {
                return C;
            }

            ReadOnlySpan<float> a = FlatReadOnlySpan(A);
            ReadOnlySpan<float> b = FlatReadOnlySpan(B);
            Span<float> c = FlatSpan(C);

            for (int i = 0; i < rowsA; i++)
            {
                int aRow = i * colsA;
                int cRow = i * rowsB;

                for (int j = 0; j < rowsB; j++)
                {
                    c[cRow + j] = Dot(a, aRow, b, j * colsB, colsA);
                }
            }

            return C;
        }

        public float[,] MatrixScale(float[,] matrix, float scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];

            ReadOnlySpan<float> src = FlatReadOnlySpan(matrix);
            Span<float> dst = FlatSpan(result);
            int length = src.Length;

            int i = 0;
            int end4 = length & ~3;
            for (; i < end4; i += 4)
            {
                dst[i] = src[i] * scalar;
                dst[i + 1] = src[i + 1] * scalar;
                dst[i + 2] = src[i + 2] * scalar;
                dst[i + 3] = src[i + 3] * scalar;
            }
            for (; i < length; i++)
            {
                dst[i] = src[i] * scalar;
            }

            return result;
        }

        public float[,] MatrixAdd(float[,] A, float[,] B)
        {
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);
            if (B.GetLength(0) != rows || B.GetLength(1) != cols)
            {
                throw new ArgumentException("Matrix dimensions must match.", nameof(B));
            }

            var result = new float[rows, cols];

            ReadOnlySpan<float> a = FlatReadOnlySpan(A);
            ReadOnlySpan<float> b = FlatReadOnlySpan(B);
            Span<float> dst = FlatSpan(result);
            int length = dst.Length;

            int i = 0;
            int end4 = length & ~3;
            for (; i < end4; i += 4)
            {
                dst[i] = a[i] + b[i];
                dst[i + 1] = a[i + 1] + b[i + 1];
                dst[i + 2] = a[i + 2] + b[i + 2];
                dst[i + 3] = a[i + 3] + b[i + 3];
            }
            for (; i < length; i++)
            {
                dst[i] = a[i] + b[i];
            }

            return result;
        }

        public float[,] MatrixAddBias(float[,] matrix, float[] bias)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];

            ReadOnlySpan<float> src = FlatReadOnlySpan(matrix);
            Span<float> dst = FlatSpan(result);

            for (int i = 0; i < rows; i++)
            {
                int row = i * cols;
                int j = 0;
                int end4 = cols & ~3;
                for (; j < end4; j += 4)
                {
                    dst[row + j] = src[row + j] + bias[j];
                    dst[row + j + 1] = src[row + j + 1] + bias[j + 1];
                    dst[row + j + 2] = src[row + j + 2] + bias[j + 2];
                    dst[row + j + 3] = src[row + j + 3] + bias[j + 3];
                }
                for (; j < cols; j++)
                {
                    dst[row + j] = src[row + j] + bias[j];
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
            if (rowCount == 0 || outputDim == 0 || inputDim == 0)
            {
                return result;
            }

            ReadOnlySpan<float> w = FlatReadOnlySpan(weights);
            ReadOnlySpan<float> x = FlatReadOnlySpan(inputMatrix);
            Span<float> y = FlatSpan(result);

            for (int i = 0; i < rowCount; i++)
            {
                int srcRow = (rowStart + i) * inputDim;
                int dstRow = i * outputDim;

                for (int j = 0; j < outputDim; j++)
                {
                    y[dstRow + j] = Dot(w, j * inputDim, x, srcRow, inputDim);
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
            if (addend.GetLength(0) != target.GetLength(0) || addend.GetLength(1) != target.GetLength(1))
            {
                throw new ArgumentException("Matrix dimensions must match.", nameof(addend));
            }

            Span<float> dst = FlatSpan(target);
            ReadOnlySpan<float> src = FlatReadOnlySpan(addend);
            int length = dst.Length;

            int i = 0;
            int end4 = length & ~3;
            for (; i < end4; i += 4)
            {
                dst[i] += src[i];
                dst[i + 1] += src[i + 1];
                dst[i + 2] += src[i + 2];
                dst[i + 3] += src[i + 3];
            }
            for (; i < length; i++)
            {
                dst[i] += src[i];
            }
        }

        public void VectorAccumulate(float[] target, float[] source)
        {
            int length = target.Length;
            int j = 0;
            int end4 = length & ~3;

            for (; j < end4; j += 4)
            {
                target[j] += source[j];
                target[j + 1] += source[j + 1];
                target[j + 2] += source[j + 2];
                target[j + 3] += source[j + 3];
            }

            for (; j < length; j++)
            {
                target[j] += source[j];
            }
        }

        #endregion

        #region Neural network
        // I realize this behaves like a transpose rather than a plain dot product-but that's exactly what we want, and it's correct.
        public float[] CalculateDotProduct(float[,] matrix, float[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (vector.Length != cols)
            {
                throw new ArgumentException($"Expected vector of length {cols}, got {vector.Length}");
            }

            var result = new float[rows];
            ReadOnlySpan<float> m = FlatReadOnlySpan(matrix);

            for (int i = 0; i < rows; i++)
            {
                result[i] = Dot(m, i * cols, vector, 0, cols);
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

            int i = 0;
            int end4 = n & ~3;
            for (; i < end4; i += 4)
            {
                grad[i] = -cost[i] * derivative[i];
                grad[i + 1] = -cost[i + 1] * derivative[i + 1];
                grad[i + 2] = -cost[i + 2] * derivative[i + 2];
                grad[i + 3] = -cost[i + 3] * derivative[i + 3];
            }
            for (; i < n; i++)
            {
                grad[i] = -cost[i] * derivative[i];
            }

            return grad;
        }

        public float[] CalculateHiddenGradients(float[,] weights, float[] nextDeltas, float[] derivative)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var delta = new float[cols];
            ReadOnlySpan<float> w = FlatReadOnlySpan(weights);

            for (int j = 0; j < cols; j++)
            {
                float sum = 0.0f;
                for (int i = 0; i < rows; i++)
                {
                    sum += w[(i * cols) + j] * nextDeltas[i];
                }

                delta[j] = sum * derivative[j];
            }

            return delta;
        }

        public float[,] UpdateWeights(float[,] weights, float[] deltas, float[] prevActivations, float learningRate, float lambda)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var updated = new float[rows, cols];

            ReadOnlySpan<float> w = FlatReadOnlySpan(weights);
            Span<float> u = FlatSpan(updated);

            for (int i = 0; i < rows; i++)
            {
                float delta = deltas[i];
                int row = i * cols;
                for (int j = 0; j < cols; j++)
                {
                    float wij = w[row + j];
                    u[row + j] = wij - learningRate * ((delta * prevActivations[j]) + (lambda * wij));
                }
            }

            return updated;
        }

        public float[] UpdateBias(float[] bias, float[] deltas, float learningRate)
        {
            int n = bias.Length;
            var updated = new float[n];

            int i = 0;
            int end4 = n & ~3;
            for (; i < end4; i += 4)
            {
                updated[i] = bias[i] - learningRate * deltas[i];
                updated[i + 1] = bias[i + 1] - learningRate * deltas[i + 1];
                updated[i + 2] = bias[i + 2] - learningRate * deltas[i + 2];
                updated[i + 3] = bias[i + 3] - learningRate * deltas[i + 3];
            }
            for (; i < n; i++)
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
            ReadOnlySpan<float> src = FlatReadOnlySpan(matrix);
            Span<float> dst = FlatSpan(result);
            ReadOnlySpan<bool> m = mask == null ? ReadOnlySpan<bool>.Empty : FlatReadOnlySpan(mask);

            if (mask == null)
            {
                for (int i = 0; i < rows; i++)
                {
                    int row = i * cols;
                    float max = float.NegativeInfinity;

                    for (int j = 0; j < cols; j++)
                    {
                        float value = src[row + j];
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
                        continue;
                    }

                    float sum = 0.0f;
                    for (int j = 0; j < cols; j++)
                    {
                        float exp = MathF.Exp(src[row + j] - max);
                        dst[row + j] = exp;
                        sum += exp;
                    }

                    if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
                    {
                        continue;
                    }

                    float invSum = 1.0f / sum;
                    for (int j = 0; j < cols; j++)
                    {
                        dst[row + j] *= invSum;
                    }
                }
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    int row = i * cols;
                    float max = float.NegativeInfinity;

                    for (int j = 0; j < cols; j++)
                    {
                        if (!m[row + j])
                        {
                            continue;
                        }

                        float value = src[row + j];
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
                        continue;
                    }

                    float sum = 0.0f;
                    for (int j = 0; j < cols; j++)
                    {
                        if (!m[row + j])
                        {
                            dst[row + j] = 0.0f;
                            continue;
                        }

                        float exp = MathF.Exp(src[row + j] - max);
                        dst[row + j] = exp;
                        sum += exp;
                    }

                    if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
                    {
                        continue;
                    }

                    float invSum = 1.0f / sum;
                    for (int j = 0; j < cols; j++)
                    {
                        dst[row + j] *= invSum;
                    }
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

            ReadOnlySpan<float> qAll = FlatReadOnlySpan(Q);
            ReadOnlySpan<float> kAll = FlatReadOnlySpan(K);
            ReadOnlySpan<float> vAll = FlatReadOnlySpan(V);
            ReadOnlySpan<float> doutAll = FlatReadOnlySpan(dConcatenated);
            ReadOnlySpan<bool> maskAll = mask == null ? ReadOnlySpan<bool>.Empty : FlatReadOnlySpan(mask);
            Span<float> dqAll = FlatSpan(dQFull);
            Span<float> dkAll = FlatSpan(dKFull);
            Span<float> dvAll = FlatSpan(dVFull);

            float[] weightsRent = ArrayPool<float>.Shared.Rent(seqLenK);
            float[] dAttnRent = ArrayPool<float>.Shared.Rent(seqLenK);
            int[] activeRent = ArrayPool<int>.Shared.Rent(seqLenK);
            float[] qRowRent = ArrayPool<float>.Shared.Rent(headDim);
            float[] doutRowRent = ArrayPool<float>.Shared.Rent(headDim);
            float[] dqRowRent = ArrayPool<float>.Shared.Rent(headDim);

            try
            {
                Span<float> weights = weightsRent.AsSpan(0, seqLenK);
                Span<float> dAttn = dAttnRent.AsSpan(0, seqLenK);
                Span<int> activeIndices = activeRent.AsSpan(0, seqLenK);
                Span<float> qRow = qRowRent.AsSpan(0, headDim);
                Span<float> doutRow = doutRowRent.AsSpan(0, headDim);
                Span<float> dqRow = dqRowRent.AsSpan(0, headDim);

                for (int head = 0; head < numHeads; head++)
                {
                    int offset = head * headDim;

                    for (int i = 0; i < seqLenQ; i++)
                    {
                        dqRow.Clear();

                        int qBase = (i * embeddingDim) + offset;
                        qAll.Slice(qBase, headDim).CopyTo(qRow);
                        doutAll.Slice(qBase, headDim).CopyTo(doutRow);

                        float maxScore = float.NegativeInfinity;
                        int activeCount = 0;

                        if (mask != null)
                        {
                            int maskRow = i * seqLenK;
                            for (int j = 0; j < seqLenK; j++)
                            {
                                if (!maskAll[maskRow + j])
                                    continue;

                                float score = Dot(qRow, 0, kAll, (j * embeddingDim) + offset, headDim) * scale;
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
                                float score = Dot(qRow, 0, kAll, (j * embeddingDim) + offset, headDim) * scale;
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

                        if (mask != null)
                        {
                            for (int n = 0; n < activeCount; n++)
                            {
                                int j = activeIndices[n];
                                float w = weights[j];
                                float dAttnJ = 0f;
                                int kvBase = (j * embeddingDim) + offset;

                                for (int k = 0; k < headDim; k++)
                                {
                                    float dout = doutRow[k];
                                    dvAll[kvBase + k] += w * dout;
                                    dAttnJ += dout * vAll[kvBase + k];
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
                                int kvBase = (j * embeddingDim) + offset;

                                for (int k = 0; k < headDim; k++)
                                {
                                    float dout = doutRow[k];
                                    dvAll[kvBase + k] += w * dout;
                                    dAttnJ += dout * vAll[kvBase + k];
                                }

                                dAttn[j] = dAttnJ;
                                rowDot += w * dAttnJ;
                            }
                        }

                        if (mask != null)
                        {
                            for (int n = 0; n < activeCount; n++)
                            {
                                int j = activeIndices[n];
                                float dDot = weights[j] * (dAttn[j] - rowDot) * scale;

                                if (dDot == 0f)
                                    continue;

                                int kvBase = (j * embeddingDim) + offset;
                                for (int k = 0; k < headDim; k++)
                                {
                                    dqRow[k] += dDot * kAll[kvBase + k];
                                    dkAll[kvBase + k] += dDot * qRow[k];
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

                                int kvBase = (j * embeddingDim) + offset;
                                for (int k = 0; k < headDim; k++)
                                {
                                    dqRow[k] += dDot * kAll[kvBase + k];
                                    dkAll[kvBase + k] += dDot * qRow[k];
                                }
                            }
                        }

                        for (int k = 0; k < headDim; k++)
                            dqAll[qBase + k] += dqRow[k];
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(weightsRent);
                ArrayPool<float>.Shared.Return(dAttnRent);
                ArrayPool<int>.Shared.Return(activeRent);
                ArrayPool<float>.Shared.Return(qRowRent);
                ArrayPool<float>.Shared.Return(doutRowRent);
                ArrayPool<float>.Shared.Return(dqRowRent);
            }

            return (dQFull, dKFull, dVFull);
        }

        #endregion

        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);
            var result = new float[batchSize, features];

            ReadOnlySpan<float> src = FlatReadOnlySpan(input);
            Span<float> dst = FlatSpan(result);

            for (int i = 0; i < batchSize; i++)
            {
                int row = i * features;
                float mean = 0.0f;

                for (int j = 0; j < features; j++)
                {
                    mean += src[row + j];
                }

                mean /= features;

                float variance = 0.0f;

                for (int j = 0; j < features; j++)
                {
                    float diff = src[row + j] - mean;
                    variance += diff * diff;
                }

                variance /= features;

                float invStd = 1.0f / MathF.Sqrt(variance + epsilon);

                for (int j = 0; j < features; j++)
                {
                    dst[row + j] = gamma[j] * ((src[row + j] - mean) * invStd) + beta[j];
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

            ReadOnlySpan<float> src = FlatReadOnlySpan(input);
            Span<float> norm = FlatSpan(normalized);
            Span<float> dst = FlatSpan(output);

            for (int i = 0; i < batchSize; i++)
            {
                int row = i * features;
                float mean = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    mean += src[row + j];
                }
                mean /= features;
                means[i] = mean;

                float variance = 0.0f;
                for (int j = 0; j < features; j++)
                {
                    float diff = src[row + j] - mean;
                    variance += diff * diff;
                }
                variance /= features;
                variances[i] = variance;

                float invStd = 1.0f / MathF.Sqrt(variance + epsilon);

                for (int j = 0; j < features; j++)
                {
                    float n = (src[row + j] - mean) * invStd;
                    norm[row + j] = n;
                    dst[row + j] = gamma[j] * n + beta[j];
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

            ReadOnlySpan<float> dout = FlatReadOnlySpan(dOut);
            ReadOnlySpan<float> norm = FlatReadOnlySpan(normalized);
            ReadOnlySpan<float> x = FlatReadOnlySpan(input);
            Span<float> dx = FlatSpan(dInput);

            for (int i = 0; i < batchSize; i++)
            {
                int row = i * features;
                float invStd = 1.0f / MathF.Sqrt(variance[i] + epsilon);
                float invStdCubed = invStd * invStd * invStd;
                float dVar = 0f;
                float dMean = 0f;
                float meanI = mean[i];

                for (int j = 0; j < features; j++)
                {
                    float dOutVal = dout[row + j];
                    dGamma[j] += dOutVal * norm[row + j];
                    dBeta[j] += dOutVal;

                    float dNorm = dOutVal * gamma[j];
                    float xMinusMean = x[row + j] - meanI;
                    dVar += dNorm * xMinusMean * (-0.5f) * invStdCubed;
                    dMean += dNorm * (-invStd);
                }

                float invN = 1.0f / features;
                float dVarScale = dVar * 2.0f * invN;
                float dMeanScale = dMean * invN;

                for (int j = 0; j < features; j++)
                {
                    float dNorm = dout[row + j] * gamma[j];
                    float xMinusMean = x[row + j] - meanI;
                    dx[row + j] = dNorm * invStd + dVarScale * xMinusMean + dMeanScale;
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
            if (seqLenQ == 0 || seqLenK == 0 || headDim == 0)
            {
                return output;
            }

            ReadOnlySpan<float> q = FlatReadOnlySpan(Q);
            ReadOnlySpan<float> kSpan = FlatReadOnlySpan(K);
            ReadOnlySpan<float> vSpan = FlatReadOnlySpan(V);
            Span<float> outSpan = FlatSpan(output);
            ReadOnlySpan<bool> m = mask == null ? ReadOnlySpan<bool>.Empty : FlatReadOnlySpan(mask);

            float[] scores = ArrayPool<float>.Shared.Rent(seqLenK);

            try
            {
                for (int head = 0; head < numHeads; head++)
                {
                    int offset = head * headDim;

                    for (int i = 0; i < seqLenQ; i++)
                    {
                        int qRow = (i * embeddingDim) + offset;
                        int outRow = (i * embeddingDim) + offset;
                        int maskRow = i * seqLenK;
                        float maxScore = float.NegativeInfinity;

                        if (mask == null)
                        {
                            for (int key = 0; key < seqLenK; key++)
                            {
                                float dot = Dot(q, qRow, kSpan, (key * embeddingDim) + offset, headDim);
                                float score = dot * scale;

                                scores[key] = score;

                                if (score > maxScore)
                                {
                                    maxScore = score;
                                }
                            }
                        }
                        else
                        {
                            for (int key = 0; key < seqLenK; key++)
                            {
                                if (!m[maskRow + key])
                                {
                                    scores[key] = float.NegativeInfinity;
                                    continue;
                                }

                                float dot = Dot(q, qRow, kSpan, (key * embeddingDim) + offset, headDim);
                                float score = dot * scale;

                                scores[key] = score;

                                if (score > maxScore)
                                {
                                    maxScore = score;
                                }
                            }
                        }

                        if (float.IsNegativeInfinity(maxScore))
                        {
                            continue;
                        }

                        float sumExp = 0f;
                        for (int key = 0; key < seqLenK; key++)
                        {
                            float score = scores[key];
                            if (float.IsNegativeInfinity(score))
                            {
                                scores[key] = 0f;
                                continue;
                            }

                            float exp = MathF.Exp(score - maxScore);
                            scores[key] = exp;
                            sumExp += exp;
                        }

                        float invSumExp = sumExp > 0f ? 1f / sumExp : 0f;

                        for (int key = 0; key < seqLenK; key++)
                        {
                            float weight = scores[key] * invSumExp;
                            if (weight == 0f)
                            {
                                continue;
                            }

                            AddScaledRow(outSpan, outRow, vSpan, (key * embeddingDim) + offset, headDim, weight);
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


        public float[,] ScaledDotProductAttention(float[,] q, float[,] k, float[,] v, int numHeads, bool[,] mask = null, bool causal = false)
        {
            if (q == null)
            {
                throw new ArgumentNullException(nameof(q));
            }
            if (k == null)
            {
                throw new ArgumentNullException(nameof(k));
            }
            if (v == null)
            {
                throw new ArgumentNullException(nameof(v));
            }
            if (numHeads <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numHeads));
            }

            int queryLen = q.GetLength(0);
            int keyLen = k.GetLength(0);
            int valueLen = v.GetLength(0);
            int embeddingDim = q.GetLength(1);

            if (queryLen <= 0)
            {
                throw new ArgumentException("Q must contain at least one row.", nameof(q));
            }
            if (keyLen <= 0)
            {
                throw new ArgumentException("K must contain at least one row.", nameof(k));
            }
            if (embeddingDim <= 0)
            {
                throw new ArgumentException("Q must contain at least one column.", nameof(q));
            }

            if (k.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("K width must match Q width.", nameof(k));
            }

            if (v.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("V width must match Q width.", nameof(v));
            }

            if (valueLen != keyLen)
            {
                throw new ArgumentException("V row count must match K row count.", nameof(v));
            }

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by numHeads.", nameof(numHeads));
            }

            if (mask != null && (mask.GetLength(0) != queryLen || mask.GetLength(1) != keyLen))
            {
                throw new ArgumentException($"Mask shape must be [{queryLen},{keyLen}], got [{mask.GetLength(0)},{mask.GetLength(1)}].", nameof(mask));
            }

            if (causal && queryLen != keyLen)
            {
                throw new ArgumentException("The simple causal path assumes queryLen == keyLen. Use an explicit mask for cross-attention or cached decoding.");
            }

            int headDim = embeddingDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);

            var output = new float[queryLen, embeddingDim];
            ReadOnlySpan<float> qSpan = FlatReadOnlySpan(q);
            ReadOnlySpan<float> kSpan = FlatReadOnlySpan(k);
            ReadOnlySpan<float> vSpan = FlatReadOnlySpan(v);
            Span<float> outSpan = FlatSpan(output);
            ReadOnlySpan<bool> m = mask == null ? ReadOnlySpan<bool>.Empty : FlatReadOnlySpan(mask);

            float[] scores = ArrayPool<float>.Shared.Rent(keyLen);

            try
            {
                for (int head = 0; head < numHeads; head++)
                {
                    int offset = head * headDim;

                    for (int qIndex = 0; qIndex < queryLen; qIndex++)
                    {
                        int usableKeyLen = causal ? Math.Min(qIndex + 1, keyLen) : keyLen;
                        int qRow = (qIndex * embeddingDim) + offset;
                        int outRow = (qIndex * embeddingDim) + offset;
                        int maskRow = qIndex * keyLen;
                        float maxScore = float.NegativeInfinity;

                        for (int keyIndex = 0; keyIndex < usableKeyLen; keyIndex++)
                        {
                            if (mask != null && !m[maskRow + keyIndex])
                            {
                                scores[keyIndex] = float.NegativeInfinity;
                                continue;
                            }

                            float dot = Dot(qSpan, qRow, kSpan, (keyIndex * embeddingDim) + offset, headDim);
                            float score = dot * scale;
                            scores[keyIndex] = score;

                            if (score > maxScore)
                            {
                                maxScore = score;
                            }
                        }

                        if (float.IsNegativeInfinity(maxScore))
                        {
                            continue;
                        }

                        float sumExp = 0.0f;

                        for (int keyIndex = 0; keyIndex < usableKeyLen; keyIndex++)
                        {
                            float score = scores[keyIndex];

                            if (float.IsNegativeInfinity(score))
                            {
                                scores[keyIndex] = 0.0f;
                                continue;
                            }

                            float exp = MathF.Exp(score - maxScore);
                            scores[keyIndex] = exp;
                            sumExp += exp;
                        }

                        if (sumExp <= 0.0f || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
                        {
                            continue;
                        }

                        float invSumExp = 1.0f / sumExp;

                        for (int keyIndex = 0; keyIndex < usableKeyLen; keyIndex++)
                        {
                            float weight = scores[keyIndex] * invSumExp;
                            if (weight == 0.0f)
                            {
                                continue;
                            }

                            AddScaledRow(outSpan, outRow, vSpan, (keyIndex * embeddingDim) + offset, headDim, weight);
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
        public float[,] FFNForwardBatch(float[,] input, int seqLen, int outputDim, Func<float[], float[]> forwardPassFn)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (forwardPassFn == null)
            {
                throw new ArgumentNullException(nameof(forwardPassFn));
            }

            if (seqLen < 0 || seqLen > input.GetLength(0))
            {
                throw new ArgumentOutOfRangeException(nameof(seqLen));
            }

            // Transformer FFNs pass NeuralNetwork.ForwardPassOnly as the delegate.
            // Use the network's stateless batched path instead of invoking the delegate
            // once per row. This removes the largest FFN CPU fallback while preserving
            // the existing IAccelerationManager API.
            if (forwardPassFn.Target is CallaghanDev.ML.NeuralNetwork network)
            {
                float[,] batchInput = seqLen == input.GetLength(0)
                    ? input
                    : SliceRows(input, 0, seqLen);

                return network.ForwardPassOnlyBatch(batchInput);
            }

            int inputDim = input.GetLength(1);
            var result = new float[seqLen, outputDim];
            int inputBytes = inputDim * sizeof(float);
            int outputBytes = outputDim * sizeof(float);

            for (int i = 0; i < seqLen; i++)
            {
                var row = new float[inputDim];
                Buffer.BlockCopy(input, i * inputBytes, row, 0, inputBytes);

                var outRow = forwardPassFn(row);
                Buffer.BlockCopy(outRow, 0, result, i * outputBytes, outputBytes);
            }

            return result;
        }
        #region Fused QKV Projection

        public (float[,] Q, float[,] K, float[,] V) ProjectQKV(float[,] input, float[,] WQ, float[] biasQ, float[,] WK, float[] biasK, float[,] WV, float[] biasV)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (WQ == null) throw new ArgumentNullException(nameof(WQ));
            if (WK == null) throw new ArgumentNullException(nameof(WK));
            if (WV == null) throw new ArgumentNullException(nameof(WV));
            if (biasQ == null) throw new ArgumentNullException(nameof(biasQ));
            if (biasK == null) throw new ArgumentNullException(nameof(biasK));
            if (biasV == null) throw new ArgumentNullException(nameof(biasV));

            int rows = input.GetLength(0);
            int inputDim = input.GetLength(1);

            int qDim = WQ.GetLength(0);
            int kDim = WK.GetLength(0);
            int vDim = WV.GetLength(0);

            if (WQ.GetLength(1) != inputDim)
                throw new ArgumentException("WQ input dimension does not match input width.", nameof(WQ));

            if (WK.GetLength(1) != inputDim)
                throw new ArgumentException("WK input dimension does not match input width.", nameof(WK));

            if (WV.GetLength(1) != inputDim)
                throw new ArgumentException("WV input dimension does not match input width.", nameof(WV));

            if (biasQ.Length != qDim)
                throw new ArgumentException("biasQ length does not match WQ output dimension.", nameof(biasQ));

            if (biasK.Length != kDim)
                throw new ArgumentException("biasK length does not match WK output dimension.", nameof(biasK));

            if (biasV.Length != vDim)
                throw new ArgumentException("biasV length does not match WV output dimension.", nameof(biasV));

            if (qDim != kDim || qDim != vDim)
                throw new ArgumentException("Q, K and V output dimensions must match.");

            var Q = new float[rows, qDim];
            var K = new float[rows, kDim];
            var V = new float[rows, vDim];

            ReadOnlySpan<float> x = FlatReadOnlySpan(input);
            ReadOnlySpan<float> wq = FlatReadOnlySpan(WQ);
            ReadOnlySpan<float> wk = FlatReadOnlySpan(WK);
            ReadOnlySpan<float> wv = FlatReadOnlySpan(WV);
            Span<float> qOut = FlatSpan(Q);
            Span<float> kOut = FlatSpan(K);
            Span<float> vOut = FlatSpan(V);

            for (int i = 0; i < rows; i++)
            {
                int inputRow = i * inputDim;
                int outputRow = i * qDim;

                for (int o = 0; o < qDim; o++)
                {
                    qOut[outputRow + o] = biasQ[o] + Dot(wq, o * inputDim, x, inputRow, inputDim);
                    kOut[outputRow + o] = biasK[o] + Dot(wk, o * inputDim, x, inputRow, inputDim);
                    vOut[outputRow + o] = biasV[o] + Dot(wv, o * inputDim, x, inputRow, inputDim);
                }
            }

            return (Q, K, V);
        }

        public float[,] BackpropQKV(float[,] input, float[,] dQ, float[,] dK, float[,] dV, float[,] WQ, float[,] WK, float[,] WV, float[,] WQGrad, float[] biasQGrad, float[,] WKGrad, float[] biasKGrad, float[,] WVGrad, float[] biasVGrad)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (dQ == null) throw new ArgumentNullException(nameof(dQ));
            if (dK == null) throw new ArgumentNullException(nameof(dK));
            if (dV == null) throw new ArgumentNullException(nameof(dV));
            if (WQ == null) throw new ArgumentNullException(nameof(WQ));
            if (WK == null) throw new ArgumentNullException(nameof(WK));
            if (WV == null) throw new ArgumentNullException(nameof(WV));
            if (WQGrad == null) throw new ArgumentNullException(nameof(WQGrad));
            if (WKGrad == null) throw new ArgumentNullException(nameof(WKGrad));
            if (WVGrad == null) throw new ArgumentNullException(nameof(WVGrad));
            if (biasQGrad == null) throw new ArgumentNullException(nameof(biasQGrad));
            if (biasKGrad == null) throw new ArgumentNullException(nameof(biasKGrad));
            if (biasVGrad == null) throw new ArgumentNullException(nameof(biasVGrad));

            int rows = input.GetLength(0);
            int inputDim = input.GetLength(1);

            int qDim = WQ.GetLength(0);
            int kDim = WK.GetLength(0);
            int vDim = WV.GetLength(0);

            if (qDim != kDim || qDim != vDim)
            {
                throw new ArgumentException("Q, K and V dimensions must match.");
            }

            int outputDim = qDim;

            if (WQ.GetLength(1) != inputDim || WK.GetLength(1) != inputDim || WV.GetLength(1) != inputDim)
            {
                throw new ArgumentException("Q/K/V weight input dimensions must match input width.");
            }

            if (dQ.GetLength(0) != rows || dK.GetLength(0) != rows || dV.GetLength(0) != rows)
            {
                throw new ArgumentException("dQ, dK and dV row counts must match input row count.");
            }

            if (dQ.GetLength(1) != outputDim || dK.GetLength(1) != outputDim || dV.GetLength(1) != outputDim)
            {
                throw new ArgumentException("dQ, dK and dV widths must match Q/K/V output dimension.");
            }

            if (WQGrad.GetLength(0) != outputDim || WQGrad.GetLength(1) != inputDim)
            {
                throw new ArgumentException("WQGrad shape mismatch.", nameof(WQGrad));
            }

            if (WKGrad.GetLength(0) != outputDim || WKGrad.GetLength(1) != inputDim)
            {
                throw new ArgumentException("WKGrad shape mismatch.", nameof(WKGrad));
            }

            if (WVGrad.GetLength(0) != outputDim || WVGrad.GetLength(1) != inputDim)
            {
                throw new ArgumentException("WVGrad shape mismatch.", nameof(WVGrad));
            }

            if (biasQGrad.Length != outputDim || biasKGrad.Length != outputDim || biasVGrad.Length != outputDim)
            {
                throw new ArgumentException("Q/K/V bias gradient lengths must match output dimension.");
            }

            var dInput = new float[rows, inputDim];

            ReadOnlySpan<float> x = FlatReadOnlySpan(input);
            ReadOnlySpan<float> dqSpan = FlatReadOnlySpan(dQ);
            ReadOnlySpan<float> dkSpan = FlatReadOnlySpan(dK);
            ReadOnlySpan<float> dvSpan = FlatReadOnlySpan(dV);
            ReadOnlySpan<float> wq = FlatReadOnlySpan(WQ);
            ReadOnlySpan<float> wk = FlatReadOnlySpan(WK);
            ReadOnlySpan<float> wv = FlatReadOnlySpan(WV);

            Span<float> wqg = FlatSpan(WQGrad);
            Span<float> wkg = FlatSpan(WKGrad);
            Span<float> wvg = FlatSpan(WVGrad);
            Span<float> dx = FlatSpan(dInput);

            for (int i = 0; i < rows; i++)
            {
                int inRow = i * inputDim;
                int gradRow = i * outputDim;

                for (int o = 0; o < outputDim; o++)
                {
                    float dq = dqSpan[gradRow + o];
                    float dk = dkSpan[gradRow + o];
                    float dv = dvSpan[gradRow + o];

                    biasQGrad[o] += dq;
                    biasKGrad[o] += dk;
                    biasVGrad[o] += dv;

                    int wRow = o * inputDim;
                    for (int d = 0; d < inputDim; d++)
                    {
                        float xv = x[inRow + d];
                        int idx = wRow + d;

                        wqg[idx] += dq * xv;
                        wkg[idx] += dk * xv;
                        wvg[idx] += dv * xv;

                        dx[inRow + d] += dq * wq[idx] + dk * wk[idx] + dv * wv[idx];
                    }
                }
            }

            return dInput;
        }

        #endregion
        #endregion

        #region Transformer training
        public void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput)
        {
            int seqLen = input.GetLength(0);
            int inDim = input.GetLength(1);
            int outDim = dOutput.GetLength(1);

            ReadOnlySpan<float> x = FlatReadOnlySpan(input);
            ReadOnlySpan<float> dy = FlatReadOnlySpan(dOutput);
            ReadOnlySpan<float> w = FlatReadOnlySpan(weights);

            Span<float> wg = FlatSpan(weightGrad);
            Span<float> dx = FlatSpan(dInput);

            for (int i = 0; i < seqLen; i++)
            {
                int xRow = i * inDim;
                int dyRow = i * outDim;

                for (int j = 0; j < outDim; j++)
                {
                    float dOutVal = dy[dyRow + j];
                    int wRow = j * inDim;

                    for (int k = 0; k < inDim; k++)
                    {
                        wg[wRow + k] += dOutVal * x[xRow + k];
                    }

                    biasGrad[j] += dOutVal;
                }

                for (int k = 0; k < inDim; k++)
                {
                    float sum = 0f;
                    for (int j = 0; j < outDim; j++)
                    {
                        sum += dy[dyRow + j] * w[(j * inDim) + k];
                    }
                    dx[xRow + k] += sum;
                }
            }
        }

        public float[,] BackpropOutputProjection(float[,] dLogits, float[,] input, float[,] weights, float[,] weightGrad, float[] biasGrad, int seqLen, int outputDim, int embeddingDim)
        {
            var dX = new float[seqLen, embeddingDim];

            ReadOnlySpan<float> dl = FlatReadOnlySpan(dLogits);
            ReadOnlySpan<float> x = FlatReadOnlySpan(input);
            ReadOnlySpan<float> w = FlatReadOnlySpan(weights);

            Span<float> wg = FlatSpan(weightGrad);
            Span<float> dx = FlatSpan(dX);

            for (int i = 0; i < seqLen; i++)
            {
                int dlRow = i * outputDim;
                int xRow = i * embeddingDim;

                for (int v = 0; v < outputDim; v++)
                {
                    float dVal = dl[dlRow + v];
                    int wRow = v * embeddingDim;

                    for (int e = 0; e < embeddingDim; e++)
                    {
                        wg[wRow + e] += x[xRow + e] * dVal;
                    }
                    biasGrad[v] += dVal;
                }

                for (int e = 0; e < embeddingDim; e++)
                {
                    float grad = 0f;

                    for (int v = 0; v < outputDim; v++)
                    {
                        grad += dl[dlRow + v] * w[(v * embeddingDim) + e];
                    }

                    dx[xRow + e] = grad;
                }
            }

            return dX;
        }

        public void BackpropInputProjection(float[,] dX, float[,] continuousInput, float[,] weightGrad, float[] biasGrad, int seqLen, int embeddingDim, int inputFeatureDim)
        {
            BackpropInputProjection(dX, continuousInput, 0, weightGrad, biasGrad, seqLen, embeddingDim, inputFeatureDim);
        }

        public void BackpropInputProjection(float[,] dX, float[,] continuousInput, int inputRowStart, float[,] weightGrad, float[] biasGrad, int seqLen, int embeddingDim, int inputFeatureDim)
        {
            if (dX == null)
            {
                throw new ArgumentNullException(nameof(dX));
            }
            if (continuousInput == null)
            {
                throw new ArgumentNullException(nameof(continuousInput));
            }
            if (weightGrad == null)
            {
                throw new ArgumentNullException(nameof(weightGrad));
            }
            if (biasGrad == null)
            {
                throw new ArgumentNullException(nameof(biasGrad));
            }

            if (inputRowStart < 0 || seqLen < 0 || inputRowStart + seqLen > continuousInput.GetLength(0))
            {
                throw new ArgumentOutOfRangeException(nameof(inputRowStart), $"Invalid input row slice: start={inputRowStart}, count={seqLen}, rows={continuousInput.GetLength(0)}.");
            }

            int continuousCols = continuousInput.GetLength(1);
            if (inputFeatureDim > continuousCols || dX.GetLength(0) < seqLen || dX.GetLength(1) < embeddingDim)
            {
                throw new ArgumentException("Input or gradient dimensions are smaller than the requested projection dimensions.");
            }

            ReadOnlySpan<float> dx = FlatReadOnlySpan(dX);
            ReadOnlySpan<float> input = FlatReadOnlySpan(continuousInput);
            Span<float> wg = FlatSpan(weightGrad);

            for (int i = 0; i < seqLen; i++)
            {
                int srcRow = (inputRowStart + i) * continuousCols;
                int dxRow = i * embeddingDim;

                for (int e = 0; e < embeddingDim; e++)
                {
                    float dVal = dx[dxRow + e];
                    int wRow = e * inputFeatureDim;

                    for (int f = 0; f < inputFeatureDim; f++)
                    {
                        wg[wRow + f] += dVal * input[srcRow + f];
                    }

                    biasGrad[e] += dVal;
                }
            }
        }

        public void AccumulateTokenEmbeddingGrad(float[,] embeddingGrad, float[,] dX, int[] tokenIds, int seqLen, int embeddingDim)
        {
            Span<float> eg = FlatSpan(embeddingGrad);
            ReadOnlySpan<float> dx = FlatReadOnlySpan(dX);

            for (int i = 0; i < seqLen; i++)
            {
                int tokenRow = tokenIds[i] * embeddingDim;
                int dxRow = i * embeddingDim;
                for (int j = 0; j < embeddingDim; j++)
                {
                    eg[tokenRow + j] += dx[dxRow + j];
                }
            }
        }

        public (float loss, float[,] dLogits) CrossEntropyLossAndGradient(float[,] logits, int[] targets, int effectiveLen)
        {
            int outputDim = logits.GetLength(1);
            float loss = 0f;
            var dLogits = new float[logits.GetLength(0), outputDim];
            float invLen = 1.0f / effectiveLen;

            ReadOnlySpan<float> l = FlatReadOnlySpan(logits);
            Span<float> dl = FlatSpan(dLogits);

            for (int i = 0; i < effectiveLen; i++)
            {
                int row = i * outputDim;
                float max = float.NegativeInfinity;
                for (int j = 0; j < outputDim; j++)
                {
                    max = Math.Max(max, l[row + j]);
                }

                float sum = 0f;
                for (int j = 0; j < outputDim; j++)
                {
                    float p = MathF.Exp(l[row + j] - max);
                    dl[row + j] = p;
                    sum += p;
                }

                float invSum = 1.0f / sum;
                int targetToken = targets[i];

                loss -= MathF.Log((dl[row + targetToken] * invSum) + 1e-10f);

                for (int j = 0; j < outputDim; j++)
                {
                    float grad = (dl[row + j] * invSum) * invLen;
                    if (j == targetToken)
                    {
                        grad -= invLen;
                    }
                    dl[row + j] = grad;
                }
            }

            loss /= effectiveLen;
            return (loss, dLogits);
        }

        public (float loss, float[,] dOutput) MSELossAndGradient(float[,] predictions, float[,] targets, int effectiveLen)
        {
            int outputDim = predictions.GetLength(1);
            if (targets.GetLength(0) < effectiveLen || targets.GetLength(1) != outputDim)
            {
                throw new ArgumentException("Target shape must match predictions for the effective length.", nameof(targets));
            }

            float loss = 0f;
            var dOutput = new float[predictions.GetLength(0), outputDim];
            float invLen = 1.0f / (effectiveLen * outputDim);

            ReadOnlySpan<float> p = FlatReadOnlySpan(predictions);
            ReadOnlySpan<float> t = FlatReadOnlySpan(targets);
            Span<float> d = FlatSpan(dOutput);

            int length = effectiveLen * outputDim;
            for (int i = 0; i < length; i++)
            {
                float diff = p[i] - t[i];
                loss += diff * diff;
                d[i] = 2.0f * diff * invLen;
            }

            loss /= (effectiveLen * outputDim);
            return (loss, dOutput);
        }

        public float MatrixSquaredNorm(float[,] matrix)
        {
            float sum = 0f;
            ReadOnlySpan<float> m = FlatReadOnlySpan(matrix);
            for (int i = 0; i < m.Length; i++)
            {
                sum += m[i] * m[i];
            }
            return sum;
        }

        public float VectorSquaredNorm(float[] vector)
        {
            float sum = 0f;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += vector[i] * vector[i];
            }
            return sum;
        }

        public void MatrixScaleInPlace(float[,] matrix, float scale)
        {
            Span<float> m = FlatSpan(matrix);
            int i = 0;
            int end4 = m.Length & ~3;
            for (; i < end4; i += 4)
            {
                m[i] *= scale;
                m[i + 1] *= scale;
                m[i + 2] *= scale;
                m[i + 3] *= scale;
            }
            for (; i < m.Length; i++)
            {
                m[i] *= scale;
            }
        }

        public void VectorScaleInPlace(float[] vector, float scale)
        {
            int i = 0;
            int end4 = vector.Length & ~3;
            for (; i < end4; i += 4)
            {
                vector[i] *= scale;
                vector[i + 1] *= scale;
                vector[i + 2] *= scale;
                vector[i + 3] *= scale;
            }
            for (; i < vector.Length; i++)
            {
                vector[i] *= scale;
            }
        }
        public void MatrixUpdate(float[,] weights, float[,] gradients, float learningRate)
        {
            if (gradients.GetLength(0) != weights.GetLength(0) || gradients.GetLength(1) != weights.GetLength(1))
            {
                throw new ArgumentException("Matrix dimensions must match.", nameof(gradients));
            }

            Span<float> w = FlatSpan(weights);
            ReadOnlySpan<float> g = FlatReadOnlySpan(gradients);

            int i = 0;
            int end4 = w.Length & ~3;

            for (; i < end4; i += 4)
            {
                w[i] -= learningRate * g[i];
                w[i + 1] -= learningRate * g[i + 1];
                w[i + 2] -= learningRate * g[i + 2];
                w[i + 3] -= learningRate * g[i + 3];
            }
            for (; i < w.Length; i++)
            {
                w[i] -= learningRate * g[i];
            }
        }

        public void VectorUpdate(float[] weights, float[] gradients, float learningRate)
        {
            int i = 0;
            int end4 = weights.Length & ~3;

            for (; i < end4; i += 4)
            {
                weights[i] -= learningRate * gradients[i];
                weights[i + 1] -= learningRate * gradients[i + 1];
                weights[i + 2] -= learningRate * gradients[i + 2];
                weights[i + 3] -= learningRate * gradients[i + 3];
            }
            for (; i < weights.Length; i++)
            {
                weights[i] -= learningRate * gradients[i];
            }
        }

        #endregion

        #region Multimodal/TACAMT/MMTAC

        public void ApplyContextTypeEmbedding(float[,] contextHidden, float[,] typeEmbedding, int[] typeIndices)
        {
            if (contextHidden == null)
            {
                throw new ArgumentNullException(nameof(contextHidden));
            }
            if (typeEmbedding == null)
            {
                throw new ArgumentNullException(nameof(typeEmbedding));
            }
            if (typeIndices == null)
            {
                throw new ArgumentNullException(nameof(typeIndices));
            }

            int rows = contextHidden.GetLength(0);
            int embDim = contextHidden.GetLength(1);

            if (typeIndices.Length != rows)
            {
                throw new ArgumentException("typeIndices length must match contextHidden row count.", nameof(typeIndices));
            }
            if (typeEmbedding.GetLength(1) != embDim)
            {
                throw new ArgumentException("typeEmbedding dimension must match contextHidden embedding dimension.", nameof(typeEmbedding));
            }

            int typeCount = typeEmbedding.GetLength(0);
            Span<float> hidden = FlatSpan(contextHidden);
            ReadOnlySpan<float> typeEmb = FlatReadOnlySpan(typeEmbedding);

            for (int i = 0; i < rows; i++)
            {
                int type = typeIndices[i];

                if ((uint)type >= (uint)typeCount)
                {
                    throw new ArgumentOutOfRangeException(nameof(typeIndices), $"Invalid context type index {type} at row {i}.");
                }

                int row = i * embDim;
                int typeRow = type * embDim;

                for (int d = 0; d < embDim; d++)
                {
                    hidden[row + d] += typeEmb[typeRow + d];
                }
            }
        }

        public float[,] ComputeTimeDiffMatrix(int priceSeqLen, float[] keyArrivalTimes)
        {
            if (priceSeqLen <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(priceSeqLen), "priceSeqLen must be positive.");
            }

            if (keyArrivalTimes == null)
            {
                throw new ArgumentNullException(nameof(keyArrivalTimes));
            }

            int numKeys = keyArrivalTimes.Length;
            var td = new float[priceSeqLen, numKeys];

            Span<float> outSpan = FlatSpan(td);

            for (int p = 0; p < priceSeqLen; p++)
            {
                int row = p * numKeys;

                for (int s = 0; s < numKeys; s++)
                {
                    outSpan[row + s] = p - keyArrivalTimes[s];
                }
            }

            return td;
        }

        public float[] ComputeMemoryAttentionScores(float[,] priceHidden, int lastPos, float[,] contextHidden, int totalCtx, float scale)
        {
            int embDim = priceHidden.GetLength(1);
            var scores = new float[totalCtx];

            ReadOnlySpan<float> price = FlatReadOnlySpan(priceHidden);
            ReadOnlySpan<float> ctx = FlatReadOnlySpan(contextHidden);

            int priceRow = lastPos * embDim;

            for (int s = 0; s < totalCtx; s++)
            {
                scores[s] = Dot(price, priceRow, ctx, s * embDim, embDim) * scale;
            }

            return scores;
        }

        public float[,] ProjectOutputBatch(float[,] hidden, float[,] outputProjection, float[] outputBias, int seqLen, int outputDim)
        {
            int embDim = hidden.GetLength(1);
            var pred = new float[seqLen, outputDim];

            ReadOnlySpan<float> h = FlatReadOnlySpan(hidden);
            ReadOnlySpan<float> proj = FlatReadOnlySpan(outputProjection);
            Span<float> p = FlatSpan(pred);

            for (int i = 0; i < seqLen; i++)
            {
                int hRow = i * embDim;
                int pRow = i * outputDim;

                for (int j = 0; j < outputDim; j++)
                {
                    p[pRow + j] = outputBias[j] + Dot(proj, j * embDim, h, hRow, embDim);
                }
            }

            return pred;
        }

        public (float[,,] decayBias, ContentAwareDecayCache cache) ContentAwareDecayForward(float[,] queryEmbeddings, float[,] keyEmbeddings, float[,] timeDiffs, float[] keyTimesFromRef, ContentAwareDecayNetwork network, bool isTraining = false, Random dropoutRng = null)
        {

            if (queryEmbeddings == null)
            {
                throw new ArgumentNullException(nameof(queryEmbeddings));
            }
            if (keyEmbeddings == null)
            {
                throw new ArgumentNullException(nameof(keyEmbeddings));
            }
            if (timeDiffs == null)
            {
                throw new ArgumentNullException(nameof(timeDiffs));
            }
            if (network == null)
            {
                throw new ArgumentNullException(nameof(network));
            }

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
            {
                throw new ArgumentException("queryEmbeddings shape does not match the decay network.", nameof(queryEmbeddings));
            }
            if (keyEmbeddings.GetLength(0) != keyLen || keyEmbeddings.GetLength(1) != contentDim)
            {
                throw new ArgumentException("keyEmbeddings shape does not match the decay network.", nameof(keyEmbeddings));
            }
            if (keyTimesFromRef != null && keyTimesFromRef.Length != keyLen)
            {
                throw new ArgumentException("keyTimesFromRef length must match key length.", nameof(keyTimesFromRef));
            }

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
                {
                    throw new ArgumentOutOfRangeException(nameof(network.MemoryAttentionDropout), "MemoryAttentionDropout must be < 1.");
                }

                cache.MemAttnDropoutMask = new float[numHeads, keyLen, keyLen];
            }

            if (useMLPDrop)
            {
                if (network.MLPDropout >= 1f)
                {
                    throw new ArgumentOutOfRangeException(nameof(network.MLPDropout), "MLPDropout must be < 1.");
                }

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
                        {
                            val += network.QueryProjection[h, p, d] * queryEmbeddings[q, d];
                        }

                        cache.QueryProj[h, q, p] = val;
                    }
                }

                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.KeyProjectionBias[h, p];

                        for (int d = 0; d < contentDim; d++)
                        {
                            val += network.KeyProjection[h, p, d] * keyEmbeddings[s, d];
                        }

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
                        {
                            val += network.TimeProj[h, p, r] * cache.TimeRawFeatures[h, s, r];
                        }

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
                        bool valid = keyTimesFromRef == null || keyTimesFromRef[j] <= keyTimesFromRef[i];

                        if (!valid)
                        {
                            scores[j] = float.NegativeInfinity;
                            continue;
                        }

                        float dot = 0f;
                        for (int p = 0; p < projDim; p++)
                        {
                            dot += cache.MemAttnQInput[h, i, p] * cache.MemAttnKInput[h, j, p];
                        }

                        scores[j] = dot * memScale;

                        if (scores[j] > maxScore)
                        {
                            maxScore = scores[j];
                        }
                    }
                    if (float.IsNegativeInfinity(maxScore))
                    {
                        continue;
                    }

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
                        {
                            cache.MemAttnWeights[h, i, j] *= inv;
                        }
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
                        {
                            val += cache.MemAttnWeights[h, i, j] * cache.KeyProj[h, j, p];
                        }

                        cache.MemAttnOutput[h, i, p] = val;
                    }
                }

                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.MemAttnOutputB[h, p];

                        for (int q = 0; q < projDim; q++)
                        {
                            val += network.MemAttnOutputW[h, p, q] * cache.MemAttnOutput[h, s, q];
                        }

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
                        {
                            cache.MLPInput[qi, si, h, idx++] = cache.QueryProj[h, qi, p];
                        }

                        for (int p = 0; p < projDim; p++)
                        {
                            cache.MLPInput[qi, si, h, idx++] = cache.RefinedKey[h, si, p];
                        }

                        for (int p = 0; p < projDim; p++)
                        {
                            cache.MLPInput[qi, si, h, idx++] = cache.QueryProj[h, qi, p] * cache.RefinedKey[h, si, p];
                        }

                        cache.MLPInput[qi, si, h, idx++] = normTd;
                        cache.MLPInput[qi, si, h, idx++] = logTd;

                        for (int j = 0; j < hiddenDim; j++)
                        {
                            float val = network.B1[h, j];

                            for (int k = 0; k < mlpInputDim; k++)
                            {
                                val += network.W1[h, j, k] * cache.MLPInput[qi, si, h, k];
                            }

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
                        {
                            logit += network.W2[h, j] * cache.MLPHidden[qi, si, h, j];
                        }

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

            ReadOnlySpan<float> qSpan = FlatReadOnlySpan(Q);
            ReadOnlySpan<float> kSpan = FlatReadOnlySpan(K);
            ReadOnlySpan<float> vSpan = FlatReadOnlySpan(V);

            Span<float> outSpan = FlatSpan(output);
            ReadOnlySpan<float> decay = decayBias == null ? ReadOnlySpan<float>.Empty : FlatReadOnlySpan(decayBias);

            for (int h = 0; h < numHeads; h++)
            {
                int offset = h * headDim;

                var scores = new float[queryLen, keyLen];
                var weights = new float[queryLen, keyLen];

                Span<float> scoreSpan = FlatSpan(scores);
                Span<float> weightSpan = FlatSpan(weights);

                for (int q = 0; q < queryLen; q++)
                {
                    int qRow = (q * embDim) + offset;
                    int scoreRow = q * keyLen;
                    float max = float.NegativeInfinity;

                    for (int s = 0; s < keyLen; s++)
                    {
                        float score = Dot(qSpan, qRow, kSpan, (s * embDim) + offset, headDim) * scale;

                        if (decayBias != null)
                        {
                            score += decay[((q * keyLen + s) * numHeads) + h];
                        }

                        scoreSpan[scoreRow + s] = score;

                        if (!float.IsNegativeInfinity(score) && score > max)
                        {
                            max = score;
                        }
                    }

                    if (float.IsNegativeInfinity(max))
                    {
                        continue;
                    }

                    float sum = 0f;
                    for (int s = 0; s < keyLen; s++)
                    {
                        float score = scoreSpan[scoreRow + s];

                        if (float.IsNegativeInfinity(score))
                        {
                            weightSpan[scoreRow + s] = 0f;
                            continue;
                        }

                        float w = MathF.Exp(score - max);
                        weightSpan[scoreRow + s] = w;
                        sum += w;
                    }

                    if (sum > 0f)
                    {
                        float inv = 1f / sum;

                        for (int s = 0; s < keyLen; s++)
                        {
                            weightSpan[scoreRow + s] *= inv;
                        }
                    }

                    int outRow = (q * embDim) + offset;

                    for (int s = 0; s < keyLen; s++)
                    {
                        float w = weightSpan[scoreRow + s];

                        if (w != 0f)
                        {
                            AddScaledRow(outSpan, outRow, vSpan, (s * embDim) + offset, headDim, w);
                        }
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
                var (bias, decayCache) = ContentAwareDecayForward(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, block.DecayNetwork, isTraining, dropoutRng);

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

            ReadOnlySpan<float> qSpan = FlatReadOnlySpan(Q);
            ReadOnlySpan<float> kSpan = FlatReadOnlySpan(K);
            ReadOnlySpan<float> vSpan = FlatReadOnlySpan(V);

            ReadOnlySpan<float> time = timeDiffs == null ? ReadOnlySpan<float>.Empty : FlatReadOnlySpan(timeDiffs);
            ReadOnlySpan<float> decay = decayBias == null ? ReadOnlySpan<float>.Empty : FlatReadOnlySpan(decayBias);

            Span<float> outSpan = FlatSpan(output);

            for (int h = 0; h < nh; h++)
            {
                int offset = h * hd;
                var scores = new float[queryLen, keyLen];
                var weights = new float[queryLen, keyLen];
                Span<float> scoreSpan = FlatSpan(scores);
                Span<float> weightSpan = FlatSpan(weights);

                for (int q = 0; q < queryLen; q++)
                {
                    int qRow = (q * ed) + offset;
                    int scoreRow = q * keyLen;
                    float max = float.NegativeInfinity;

                    for (int s = 0; s < keyLen; s++)
                    {
                        bool valid = timeDiffs == null || time[scoreRow + s] >= 0f;

                        if (!valid)
                        {
                            scoreSpan[scoreRow + s] = float.NegativeInfinity;
                            continue;
                        }

                        float score = Dot(qSpan, qRow, kSpan, (s * ed) + offset, hd) * scale;

                        if (decayBias != null)
                        {
                            score += decay[((q * keyLen + s) * nh) + h];
                        }

                        scoreSpan[scoreRow + s] = score;
                        if (!float.IsNegativeInfinity(score) && score > max)
                        {
                            max = score;
                        }
                    }

                    if (float.IsNegativeInfinity(max))
                        continue;

                    float sum = 0f;
                    for (int s = 0; s < keyLen; s++)
                    {
                        float score = scoreSpan[scoreRow + s];
                        if (float.IsNegativeInfinity(score))
                        {
                            weightSpan[scoreRow + s] = 0f;
                            continue;
                        }

                        float w = MathF.Exp(score - max);
                        weightSpan[scoreRow + s] = w;
                        sum += w;
                    }

                    if (sum > 0f)
                    {
                        float inv = 1f / sum;
                        for (int s = 0; s < keyLen; s++)
                            weightSpan[scoreRow + s] *= inv;
                    }

                    int outRow = (q * ed) + offset;
                    for (int s = 0; s < keyLen; s++)
                    {
                        float w = weightSpan[scoreRow + s];
                        if (w != 0f)
                            AddScaledRow(outSpan, outRow, vSpan, (s * ed) + offset, hd, w);
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
            Span<float> m = FlatSpan(matrix);
            int i = 0;
            int end4 = m.Length & ~3;
            for (; i < end4; i += 4)
            {
                m[i] *= scale;
                m[i + 1] *= scale;
                m[i + 2] *= scale;
                m[i + 3] *= scale;
            }
            for (; i < m.Length; i++)
            {
                m[i] *= scale;
            }
        }

        public float MatrixSquaredNorm3D(float[,,] matrix)
        {
            float sum = 0f;

            ReadOnlySpan<float> m = FlatReadOnlySpan(matrix);

            for (int i = 0; i < m.Length; i++)
            {
                sum += m[i] * m[i];
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
            ReadOnlySpan<float> proj = FlatReadOnlySpan(projection);

            for (int d = 0; d < ed; d++)
            {
                output[d] = bias[d] + Dot(proj, d * gd, globalFeatures, 0, gd);
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
            int bytesPerRow = embeddingDim * sizeof(float);

            for (int i = 0; i < tokenIds.Length; i++)
            {
                int tokenId = tokenIds[i];

                if ((uint)tokenId >= (uint)vocabSize)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token out of range: {tokenId}");
                }

                Buffer.BlockCopy(embedding, tokenId * bytesPerRow, output, i * bytesPerRow, bytesPerRow);
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

            ReadOnlySpan<float> m = FlatReadOnlySpan(matrix);

            for (int r = 0; r < rows; r++)
            {
                int row = r * cols;
                for (int d = 0; d < cols; d++)
                {
                    output[d] += m[row + d];
                }
            }

            float inv = 1.0f / rows;
            for (int d = 0; d < cols; d++)
            {
                output[d] *= inv;
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
            Span<float> ctx = FlatSpan(contextHidden);
            ReadOnlySpan<float> typeEmb = FlatReadOnlySpan(contextTypeEmbedding);
            ReadOnlySpan<float> news = newsHidden == null ? ReadOnlySpan<float>.Empty : FlatReadOnlySpan(newsHidden);

            int row = 0;

            if (globalToken != null)
            {
                int dst = row * ed;
                int typeRow = 2 * ed;
                for (int d = 0; d < ed; d++)
                {
                    ctx[dst + d] = globalToken[d] + typeEmb[typeRow + d];
                }

                contextTimes[row] = 0f;
                row++;
            }

            for (int i = 0; i < numNews; i++)
            {
                int dst = row * ed;
                int src = i * ed;
                for (int d = 0; d < ed; d++)
                {
                    ctx[dst + d] = news[src + d] + typeEmb[d];
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
            var quality = new float[sl, 1];
            var qualityLogits = new float[sl];
            var direction = new float[sl, 1];
            var midDirection = new float[sl, 1];
            float[,] confidence = useConfidenceHead ? new float[sl, 1] : null;

            ReadOnlySpan<float> h = FlatReadOnlySpan(hidden);
            ReadOnlySpan<float> raw = FlatReadOnlySpan(rawRegression);
            ReadOnlySpan<float> rangeProj = FlatReadOnlySpan(rangeProjection);
            ReadOnlySpan<float> qualityProj = FlatReadOnlySpan(qualityProjection);
            ReadOnlySpan<float> directionProj = FlatReadOnlySpan(directionProjection);
            ReadOnlySpan<float> midDirectionProj = FlatReadOnlySpan(midDirectionProjection);
            ReadOnlySpan<float> confidenceProj = confidenceProjection == null ? ReadOnlySpan<float>.Empty : FlatReadOnlySpan(confidenceProjection);
            Span<float> reg = FlatSpan(regression);
            Span<float> rangeSpan = FlatSpan(range);
            Span<float> qualitySpan = FlatSpan(quality);
            Span<float> directionSpan = FlatSpan(direction);
            Span<float> midDirectionSpan = FlatSpan(midDirection);
            Span<float> confidenceSpan = confidence == null ? Span<float>.Empty : FlatSpan(confidence);

            for (int t = 0; t < sl; t++)
            {
                int hRow = t * ed;
                int rRow = t * rDim;

                float upBase = Softplus(raw[rRow]);
                float downBase = Softplus(raw[rRow + 1]);
                float rangeLogit = rangeBias[0] + Dot(rangeProj, 0, h, hRow, ed);
                rangeLogits[t] = rangeLogit;

                float rangeValue = Softplus(rangeLogit);
                float den = upBase + downBase;
                float upShare = den > 1e-6f ? upBase / den : 0.5f;
                float downShare = 1f - upShare;
                float close = raw[rRow + 2];

                reg[rRow] = close + rangeValue * upShare;
                reg[rRow + 1] = close - rangeValue * downShare;
                reg[rRow + 2] = close;
                rangeSpan[t] = rangeValue;

                float qualityLogit = qualityBias[0] + Dot(qualityProj, 0, h, hRow, ed);
                qualityLogits[t] = qualityLogit;
                qualitySpan[t] = StableSigmoid(qualityLogit);

                float directionLogit = directionBias[0] + Dot(directionProj, 0, h, hRow, ed);
                directionSpan[t] = StableSigmoid(directionLogit);

                float midDirectionLogit = midDirectionBias[0] + Dot(midDirectionProj, 0, h, hRow, ed);
                midDirectionSpan[t] = StableSigmoid(midDirectionLogit);

                if (useConfidenceHead)
                {
                    float confidenceLogit = confidenceBias[0] + Dot(confidenceProj, 0, h, hRow, ed);
                    confidenceSpan[t] = StableSigmoid(confidenceLogit);
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
                float value = MathF.Exp(scores[i] - max);
                output[i] = value;
                sum += value;
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
            if (q == null)
            {
                throw new ArgumentNullException(nameof(q));
            }
            if (k == null)
            {
                throw new ArgumentNullException(nameof(k));
            }
            if (v == null)
            {
                throw new ArgumentNullException(nameof(v));
            }
            if (dOutput == null)
            {
                throw new ArgumentNullException(nameof(dOutput));
            }
            if (attentionWeights == null)
            {
                throw new ArgumentNullException(nameof(attentionWeights));
            }

            int queryLen = q.GetLength(0);
            int keyLen = k.GetLength(0);
            int headDim = embeddingDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);

            var dQ = new float[queryLen, embeddingDim];
            var dK = new float[keyLen, embeddingDim];
            var dV = new float[keyLen, embeddingDim];

            float[,,] dDecayBias = timeDiffs != null ? new float[queryLen, keyLen, numHeads] : null;

            ReadOnlySpan<float> qSpan = FlatReadOnlySpan(q);
            ReadOnlySpan<float> kSpan = FlatReadOnlySpan(k);
            ReadOnlySpan<float> vSpan = FlatReadOnlySpan(v);
            ReadOnlySpan<float> dOut = FlatReadOnlySpan(dOutput);

            Span<float> dQSpan = FlatSpan(dQ);
            Span<float> dKSpan = FlatSpan(dK);
            Span<float> dVSpan = FlatSpan(dV);

            Span<float> dDecay = dDecayBias == null ? Span<float>.Empty : FlatSpan(dDecayBias);

            float[] dWeights = ArrayPool<float>.Shared.Rent(keyLen);
            try
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int offset = h * headDim;
                    var weights = attentionWeights[h];
                    ReadOnlySpan<float> w = FlatReadOnlySpan(weights);

                    for (int p = 0; p < queryLen; p++)
                    {
                        int weightRow = p * keyLen;
                        int dOutRow = (p * embeddingDim) + offset;

                        for (int s = 0; s < keyLen; s++)
                        {
                            float dot = 0f;
                            float weight = w[weightRow + s];
                            int keyRow = (s * embeddingDim) + offset;

                            for (int d = 0; d < headDim; d++)
                            {
                                float dout = dOut[dOutRow + d];
                                dVSpan[keyRow + d] += weight * dout;
                                dot += dout * vSpan[keyRow + d];
                            }

                            dWeights[s] = dot;
                        }

                        float dotWeightGrad = 0f;

                        for (int s = 0; s < keyLen; s++)
                        {
                            dotWeightGrad += w[weightRow + s] * dWeights[s];
                        }

                        for (int s = 0; s < keyLen; s++)
                        {
                            float dScore = w[weightRow + s] * (dWeights[s] - dotWeightGrad);
                            float dScaled = dScore * scale;
                            int keyRow = (s * embeddingDim) + offset;
                            int qRow = (p * embeddingDim) + offset;

                            for (int d = 0; d < headDim; d++)
                            {
                                dQSpan[qRow + d] += dScaled * kSpan[keyRow + d];
                                dKSpan[keyRow + d] += dScaled * qSpan[qRow + d];
                            }

                            if (dDecayBias != null)
                            {
                                dDecay[((p * keyLen + s) * numHeads) + h] += dScore;
                            }
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(dWeights);
            }

            return (dQ, dK, dV, dDecayBias);
        }
        public (float[,] contextHidden, float[] contextTimes, int numGlobal, int numNews, int numPrice) BuildMmtacContextWithPrice(float[,] newsHidden, float[] newsTimes, float[] globalToken, float[,] priceContextHidden, float[] priceContextTimes, float[,] contextTypeEmbedding)
        {
            if (contextTypeEmbedding == null)
            {
                throw new ArgumentNullException(nameof(contextTypeEmbedding));
            }

            int ed = contextTypeEmbedding.GetLength(1);
            int numGlobal = globalToken != null ? 1 : 0;
            int numNews = newsHidden != null ? newsHidden.GetLength(0) : 0;
            int numPrice = priceContextHidden != null ? priceContextHidden.GetLength(0) : 0;
            int total = numGlobal + numNews + numPrice;

            if (total == 0)
            {
                return (null, null, 0, 0, 0);
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

            if (priceContextHidden != null && priceContextHidden.GetLength(1) != ed)
            {
                throw new ArgumentException("priceContextHidden embedding dimension mismatch.", nameof(priceContextHidden));
            }

            if (priceContextTimes != null && priceContextTimes.Length != numPrice)
            {
                throw new ArgumentException("priceContextTimes length must match priceContextHidden row count.", nameof(priceContextTimes));
            }

            var contextHidden = new float[total, ed];
            var contextTimes = new float[total];
            Span<float> ctx = FlatSpan(contextHidden);
            ReadOnlySpan<float> typeEmb = FlatReadOnlySpan(contextTypeEmbedding);
            ReadOnlySpan<float> news = newsHidden == null ? ReadOnlySpan<float>.Empty : FlatReadOnlySpan(newsHidden);
            ReadOnlySpan<float> price = priceContextHidden == null ? ReadOnlySpan<float>.Empty : FlatReadOnlySpan(priceContextHidden);
            int row = 0;

            if (globalToken != null)
            {
                int dst = row * ed;
                int typeRow = 2 * ed;
                for (int d = 0; d < ed; d++)
                {
                    ctx[dst + d] = globalToken[d] + typeEmb[typeRow + d];
                }

                contextTimes[row] = 0f;
                row++;
            }

            for (int i = 0; i < numNews; i++)
            {
                int dst = row * ed;
                int src = i * ed;
                for (int d = 0; d < ed; d++)
                {
                    ctx[dst + d] = news[src + d] + typeEmb[d];
                }

                contextTimes[row] = newsTimes != null ? newsTimes[i] : 0f;
                row++;
            }

            int priceTypeRow = ed;
            for (int i = 0; i < numPrice; i++)
            {
                int dst = row * ed;
                int src = i * ed;

                for (int d = 0; d < ed; d++)
                {
                    ctx[dst + d] = price[src + d] + typeEmb[priceTypeRow + d];
                }

                contextTimes[row] = priceContextTimes != null ? priceContextTimes[i] : 0f;
                row++;
            }

            return (contextHidden, contextTimes, numGlobal, numNews, numPrice);
        }
        public (float loss, float[,] dHidden) BackpropMmtacOutputHeads(float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence, float[,] targetRegression, float[,] targetRange, float[,] targetQuality, float[,] targetDirection, float[,] targetMidDirection, float[] previousClose, float[] confidenceTargets, float[,] hidden, float[,] regressionLogits, float[] rangeLogits, float[,] regressionProjection, float[,] rangeProjection, float[,] qualityProjection, float[,] directionProjection, float[,] midDirectionProjection, float[,] confidenceProjection, float[,] regressionProjectionGrad, float[] regressionBiasGrad, float[,] rangeProjectionGrad, float[] rangeBiasGrad, float[,] qualityProjectionGrad, float[] qualityBiasGrad, float[,] directionProjectionGrad, float[] directionBiasGrad, float[,] midDirectionProjectionGrad, float[] midDirectionBiasGrad, float[,] confidenceProjectionGrad, float[] confidenceBiasGrad, float rangeLossWeight, float qualityLossWeight, float directionLossWeight, float midDirectionLossWeight, float closeDirectionConsistencyWeight, float closeDirectionConsistencyMargin, float confidenceLossWeight, bool useConfidenceHead)
        {
            if (regression == null) throw new ArgumentNullException(nameof(regression));
            if (targetRegression == null) throw new ArgumentNullException(nameof(targetRegression));
            if (hidden == null) throw new ArgumentNullException(nameof(hidden));
            if (regressionLogits == null) throw new ArgumentNullException(nameof(regressionLogits));
            if (rangeLogits == null) throw new ArgumentNullException(nameof(rangeLogits));

            int sl = regression.GetLength(0);
            int rDim = regression.GetLength(1);
            int ed = hidden.GetLength(1);

            if (rDim < 3)
            {
                throw new ArgumentException("Regression output must contain high, low and close columns.", nameof(regression));
            }
            if (hidden.GetLength(0) != sl)
            {
                throw new ArgumentException("hidden row count must match output row count.", nameof(hidden));
            }
            if (confidenceTargets != null && confidenceTargets.Length < sl)
            {
                throw new ArgumentException("confidenceTargets length must be at least the sequence length.", nameof(confidenceTargets));
            }

            float mseLoss = 0f;
            var dHigh = new float[sl];
            var dLow = new float[sl];
            var dClose = new float[sl];
            var dRangeOutput = new float[sl];
            float invRegCount = 1.0f / (sl * rDim);

            for (int t = 0; t < sl; t++)
            {
                float diffHigh = regression[t, 0] - targetRegression[t, 0];
                float diffLow = regression[t, 1] - targetRegression[t, 1];
                float diffClose = regression[t, 2] - targetRegression[t, 2];

                mseLoss += diffHigh * diffHigh + diffLow * diffLow + diffClose * diffClose;
                dHigh[t] = 2f * diffHigh * invRegCount;
                dLow[t] = 2f * diffLow * invRegCount;
                dClose[t] = 2f * diffClose * invRegCount;
            }

            mseLoss *= invRegCount;

            float closeDirectionLoss = 0f;
            if (closeDirectionConsistencyWeight > 0f && previousClose != null)
            {
                for (int t = 0; t < sl; t++)
                {
                    float sign = targetDirection[t, 0] >= 0.5f ? 1f : -1f;
                    float z = sign * (regression[t, 2] - previousClose[t] - sign * closeDirectionConsistencyMargin);

                    if (z > 20f)
                    {
                        closeDirectionLoss += MathF.Exp(-z);
                    }
                    else if (z < -20f)
                    {
                        closeDirectionLoss += -z;
                    }
                    else
                    {
                        closeDirectionLoss += MathF.Log(1f + MathF.Exp(-z));
                    }

                    float sigmoidNegZ;
                    if (z >= 0f)
                    {
                        float ez = MathF.Exp(-z);
                        sigmoidNegZ = ez / (1f + ez);
                    }
                    else
                    {
                        float ez = MathF.Exp(z);
                        sigmoidNegZ = 1f / (1f + ez);
                    }

                    dClose[t] += -sign * sigmoidNegZ * closeDirectionConsistencyWeight / sl;
                }

                closeDirectionLoss /= sl;
            }

            float rangeLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                float diff = range[t, 0] - targetRange[t, 0];
                rangeLoss += diff * diff;
                dRangeOutput[t] = 2f * diff / sl * rangeLossWeight;
            }
            rangeLoss /= sl;

            var dRawRegression = new float[sl, rDim];
            var dHidden = new float[sl, ed];

            for (int t = 0; t < sl; t++)
            {
                float upLogit = regressionLogits[t, 0];
                float downLogit = regressionLogits[t, 1];
                float rangeLogit = rangeLogits[t];

                float upBase = Softplus(upLogit);
                float downBase = Softplus(downLogit);
                float den = upBase + downBase;
                float upShare = den > 1e-6f ? upBase / den : 0.5f;
                float downShare = 1f - upShare;
                float rangeValue = Softplus(rangeLogit);

                float gHigh = dHigh[t];
                float gLow = dLow[t];
                float gClose = dClose[t];
                float gRange = dRangeOutput[t];

                float dCloseRaw = gHigh + gLow + gClose;
                float dRangeValue = gHigh * upShare - gLow * downShare + gRange;
                float dShare = rangeValue * (gHigh + gLow);

                float dUpBase = 0f;
                float dDownBase = 0f;
                if (den > 1e-6f)
                {
                    float invDenSq = 1f / (den * den);
                    dUpBase = dShare * downBase * invDenSq;
                    dDownBase = -dShare * upBase * invDenSq;
                }

                dRawRegression[t, 0] = dUpBase * StableSigmoid(upLogit);
                dRawRegression[t, 1] = dDownBase * StableSigmoid(downLogit);
                dRawRegression[t, 2] = dCloseRaw;

                float dRangeLogit = dRangeValue * StableSigmoid(rangeLogit);
                rangeBiasGrad[0] += dRangeLogit;

                for (int k = 0; k < ed; k++)
                {
                    rangeProjectionGrad[0, k] += dRangeLogit * hidden[t, k];
                    dHidden[t, k] += dRangeLogit * rangeProjection[0, k];
                }
            }

            for (int t = 0; t < sl; t++)
            {
                for (int v = 0; v < rDim; v++)
                {
                    float dVal = dRawRegression[t, v];
                    regressionBiasGrad[v] += dVal;

                    for (int k = 0; k < ed; k++)
                    {
                        regressionProjectionGrad[v, k] += dVal * hidden[t, k];
                        dHidden[t, k] += dVal * regressionProjection[v, k];
                    }
                }
            }

            float qualityLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                float p = quality[t, 0];
                float diff = p - targetQuality[t, 0];
                qualityLoss += diff * diff;

                float dOutput = 2f * diff / sl * qualityLossWeight;
                float dLogit = dOutput * p * (1f - p);
                qualityBiasGrad[0] += dLogit;

                for (int k = 0; k < ed; k++)
                {
                    qualityProjectionGrad[0, k] += dLogit * hidden[t, k];
                    dHidden[t, k] += dLogit * qualityProjection[0, k];
                }
            }
            qualityLoss /= sl;

            float directionLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                float p = direction[t, 0];
                float y = targetDirection[t, 0];
                float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
                directionLoss -= y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc);

                float dLogit = (p - y) * directionLossWeight / sl;
                directionBiasGrad[0] += dLogit;

                for (int k = 0; k < ed; k++)
                {
                    directionProjectionGrad[0, k] += dLogit * hidden[t, k];
                    dHidden[t, k] += dLogit * directionProjection[0, k];
                }
            }
            directionLoss /= sl;

            float midDirectionLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                float p = midDirection[t, 0];
                float y = targetMidDirection[t, 0];
                float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);

                midDirectionLoss -= y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc);

                float dLogit = (p - y) * midDirectionLossWeight / sl;
                midDirectionBiasGrad[0] += dLogit;

                for (int k = 0; k < ed; k++)
                {
                    midDirectionProjectionGrad[0, k] += dLogit * hidden[t, k];
                    dHidden[t, k] += dLogit * midDirectionProjection[0, k];
                }
            }
            midDirectionLoss /= sl;

            float confidenceLoss = 0f;
            float effectiveConfidenceWeight = useConfidenceHead ? MathF.Max(0f, confidenceLossWeight) : 0f;

            if (useConfidenceHead && confidence != null && confidenceProjection != null && confidenceProjectionGrad != null && confidenceBiasGrad != null && effectiveConfidenceWeight > 0f)
            {
                for (int t = 0; t < sl; t++)
                {
                    float p = confidence[t, 0];
                    float y;

                    if (confidenceTargets != null)
                    {
                        y = confidenceTargets[t];
                    }
                    else
                    {
                        float sq = 0f;

                        for (int j = 0; j < rDim; j++)
                        {
                            float diff = regression[t, j] - targetRegression[t, j];
                            sq += diff * diff;
                        }

                        y = MathF.Exp(-5f * MathF.Sqrt(sq / rDim));
                    }

                    float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
                    confidenceLoss -= y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc);

                    float dLogit = (p - y) * effectiveConfidenceWeight / sl;
                    confidenceBiasGrad[0] += dLogit;

                    for (int k = 0; k < ed; k++)
                    {
                        confidenceProjectionGrad[0, k] += dLogit * hidden[t, k];
                        dHidden[t, k] += dLogit * confidenceProjection[0, k];
                    }
                }

                confidenceLoss /= sl;
            }

            float loss = mseLoss + closeDirectionConsistencyWeight * closeDirectionLoss + rangeLossWeight * rangeLoss + qualityLossWeight * qualityLoss + directionLossWeight * directionLoss + midDirectionLossWeight * midDirectionLoss + effectiveConfidenceWeight * confidenceLoss;

            return (loss, dHidden);
        }
        public void AccumulateMmtacContextGradients(float[,] dContextA, float[,] dContextB, float[,] contextTypeEmbeddingGrad, float[,] dLiveNewsHidden, float[] dGlobalHidden, int numGlobal, int numStoredNews, int numNews, int numLiveNews, int numPriceContext, int totalContext, int priceOffset)
        {
            if (contextTypeEmbeddingGrad == null)
            {
                throw new ArgumentNullException(nameof(contextTypeEmbeddingGrad));
            }
            if (dContextA == null && dContextB == null)
            {
                return;
            } 

            int ed = contextTypeEmbeddingGrad.GetLength(1);

            float GetGrad(int row, int col)
            {
                float g = 0f;

                if (dContextA != null && row < dContextA.GetLength(0))
                {
                    g += dContextA[row, col];
                }
                if (dContextB != null && row < dContextB.GetLength(0))
                {
                    g += dContextB[row, col];
                }

                return g;
            }

            for (int gi = 0; gi < numGlobal && gi < totalContext; gi++)
            {
                for (int j = 0; j < ed; j++)
                {
                    float g = GetGrad(gi, j);

                    contextTypeEmbeddingGrad[2, j] += g;

                    if (dGlobalHidden != null)
                    {
                        dGlobalHidden[j] += g;
                    }
                }
            }

            for (int i = 0; i < numNews; i++)
            {
                int ctxIdx = numGlobal + i;
                if (ctxIdx >= totalContext)
                {
                    break;
                }

                bool isLive = i >= numStoredNews;
                int liveIdx = i - numStoredNews;

                for (int j = 0; j < ed; j++)
                {
                    float g = GetGrad(ctxIdx, j);
                    contextTypeEmbeddingGrad[0, j] += g;

                    if (isLive && dLiveNewsHidden != null && liveIdx >= 0 && liveIdx < numLiveNews)
                    {
                        dLiveNewsHidden[liveIdx, j] += g;
                    }
                }
            }

            for (int i = 0; i < numPriceContext; i++)
            {
                int ctxIdx = priceOffset + i;

                if (ctxIdx >= totalContext)
                {
                    break;
                }

                for (int j = 0; j < ed; j++)
                {
                    contextTypeEmbeddingGrad[1, j] += GetGrad(ctxIdx, j);
                }
            }
        }
        public void AccumulateGlobalProjectionGradients(float[] dGlobalHidden, float[] globalFeatures, float[,] projectionGrad, float[] biasGrad)
        {
            if (dGlobalHidden == null || globalFeatures == null || projectionGrad == null || biasGrad == null)
            {
                return;
            }

            int ed = dGlobalHidden.Length;
            int gd = globalFeatures.Length;
            Span<float> pg = FlatSpan(projectionGrad);

            for (int d = 0; d < ed; d++)
            {
                float gToken = dGlobalHidden[d];

                biasGrad[d] += gToken;

                int row = d * gd;

                for (int g = 0; g < gd; g++)
                {
                    pg[row + g] += gToken * globalFeatures[g];
                }
            }
        }
        public float[,] ExpandMeanPoolGradient(float[,] pooledGradient, int rowIndex, int rowCount, int embeddingDim)
        {
            if (pooledGradient == null)
            {
                throw new ArgumentNullException(nameof(pooledGradient));
            }
            if (rowCount <= 0)
            {
                return new float[0, embeddingDim];
            }
            if (rowIndex < 0 || rowIndex >= pooledGradient.GetLength(0))
            {
                throw new ArgumentOutOfRangeException(nameof(rowIndex));
            }

            var result = new float[rowCount, embeddingDim];

            float inv = 1.0f / rowCount;

            ReadOnlySpan<float> pooled = FlatReadOnlySpan(pooledGradient);

            Span<float> dst = FlatSpan(result);

            int pooledCols = pooledGradient.GetLength(1);

            int src = rowIndex * pooledCols;

            for (int t = 0; t < rowCount; t++)
            {
                int row = t * embeddingDim;
                for (int d = 0; d < embeddingDim; d++)
                {
                    dst[row + d] = pooled[src + d] * inv;
                }
            }

            return result;
        }


        public (float[,] K, float[,] V) ProjectKV(float[,] input, float[,] WK, float[] biasK, float[,] WV, float[] biasV)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (WK == null) throw new ArgumentNullException(nameof(WK));
            if (WV == null) throw new ArgumentNullException(nameof(WV));
            if (biasK == null) throw new ArgumentNullException(nameof(biasK));
            if (biasV == null) throw new ArgumentNullException(nameof(biasV));

            int rows = input.GetLength(0);
            int inputDim = input.GetLength(1);
            int kDim = WK.GetLength(0);
            int vDim = WV.GetLength(0);

            if (kDim != vDim)
            {
                throw new ArgumentException("K and V output dimensions must match.");
            }
            if (WK.GetLength(1) != inputDim)
            {
                throw new ArgumentException("WK input dimension does not match input width.", nameof(WK));
            }
            if (WV.GetLength(1) != inputDim)
            {
                throw new ArgumentException("WV input dimension does not match input width.", nameof(WV));
            }
            if (biasK.Length != kDim)
            {
                throw new ArgumentException("biasK length does not match WK output dimension.", nameof(biasK));
            }
            if (biasV.Length != vDim)
            {
                throw new ArgumentException("biasV length does not match WV output dimension.", nameof(biasV));
            }

            var K = new float[rows, kDim];
            var V = new float[rows, vDim];

            ReadOnlySpan<float> x = FlatReadOnlySpan(input);
            ReadOnlySpan<float> wk = FlatReadOnlySpan(WK);
            ReadOnlySpan<float> wv = FlatReadOnlySpan(WV);

            Span<float> kOut = FlatSpan(K);
            Span<float> vOut = FlatSpan(V);

            for (int i = 0; i < rows; i++)
            {
                int xRow = i * inputDim;
                int outRow = i * kDim;

                for (int o = 0; o < kDim; o++)
                {
                    kOut[outRow + o] = biasK[o] + Dot(wk, o * inputDim, x, xRow, inputDim);
                    vOut[outRow + o] = biasV[o] + Dot(wv, o * inputDim, x, xRow, inputDim);
                }
            }

            return (K, V);
        }
        public float[,] BackpropKV(float[,] input, float[,] dK, float[,] dV, float[,] WK, float[,] WV, float[,] WKGrad, float[] biasKGrad, float[,] WVGrad, float[] biasVGrad)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }
            if (dK == null)
            {
                throw new ArgumentNullException(nameof(dK));
            }
            if (dV == null)
            {
                throw new ArgumentNullException(nameof(dV));
            }
            if (WK == null)
            {
                throw new ArgumentNullException(nameof(WK));
            }
            if (WV == null)
            {
                throw new ArgumentNullException(nameof(WV));
            }
            if (WKGrad == null)
            {
                throw new ArgumentNullException(nameof(WKGrad));
            }
            if (WVGrad == null)
            {
                throw new ArgumentNullException(nameof(WVGrad));
            }
            if (biasKGrad == null)
            {
                throw new ArgumentNullException(nameof(biasKGrad));
            }
            if (biasVGrad == null)
            {
                throw new ArgumentNullException(nameof(biasVGrad));
            }

            int rows = input.GetLength(0);
            int inputDim = input.GetLength(1);
            int kDim = WK.GetLength(0);
            int vDim = WV.GetLength(0);

            if (kDim != vDim)
            {
                throw new ArgumentException("K and V dimensions must match.");
            }

            int outputDim = kDim;

            if (WK.GetLength(1) != inputDim || WV.GetLength(1) != inputDim)
            {
                throw new ArgumentException("K/V weight input dimensions must match input width.");
            }
            if (dK.GetLength(0) != rows || dV.GetLength(0) != rows)
            {
                throw new ArgumentException("dK and dV row counts must match input row count.");
            }
            if (dK.GetLength(1) != outputDim || dV.GetLength(1) != outputDim)
            {
                throw new ArgumentException("dK and dV widths must match K/V output dimension.");
            }
            if (WKGrad.GetLength(0) != outputDim || WKGrad.GetLength(1) != inputDim)
            {
                throw new ArgumentException("WKGrad shape mismatch.", nameof(WKGrad));
            }
            if (WVGrad.GetLength(0) != outputDim || WVGrad.GetLength(1) != inputDim)
            {
                throw new ArgumentException("WVGrad shape mismatch.", nameof(WVGrad));
            }
            if (biasKGrad.Length != outputDim || biasVGrad.Length != outputDim)
            {
                throw new ArgumentException("K/V bias gradient lengths must match output dimension.");
            }

            var dInput = new float[rows, inputDim];

            ReadOnlySpan<float> x = FlatReadOnlySpan(input);
            ReadOnlySpan<float> dkSpan = FlatReadOnlySpan(dK);
            ReadOnlySpan<float> dvSpan = FlatReadOnlySpan(dV);
            ReadOnlySpan<float> wk = FlatReadOnlySpan(WK);
            ReadOnlySpan<float> wv = FlatReadOnlySpan(WV);

            Span<float> wkg = FlatSpan(WKGrad);
            Span<float> wvg = FlatSpan(WVGrad);
            Span<float> dx = FlatSpan(dInput);

            for (int i = 0; i < rows; i++)
            {
                int inputRow = i * inputDim;
                int gradRow = i * outputDim;

                for (int o = 0; o < outputDim; o++)
                {
                    float dk = dkSpan[gradRow + o];
                    float dv = dvSpan[gradRow + o];

                    biasKGrad[o] += dk;
                    biasVGrad[o] += dv;

                    int wRow = o * inputDim;

                    for (int d = 0; d < inputDim; d++)
                    {
                        float xv = x[inputRow + d];
                        int idx = wRow + d;

                        wkg[idx] += dk * xv;
                        wvg[idx] += dv * xv;

                        dx[inputRow + d] += dk * wk[idx] + dv * wv[idx];
                    }
                }
            }

            return dInput;
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

            var bestPair = pairCounts.Where(kv => kv.Value >= minFrequency).OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key.left + kv.Key.right).FirstOrDefault();

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
                Array.Fill(padded, padTokenId, tokenIds.Length, maxLength - tokenIds.Length);
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

                ApplyRotaryPositionEmbeddingHeadCoreInPlace(matrix, startCol, headDim, baseTheta, inverse);
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

            ApplyRotaryPositionEmbeddingHeadCoreInPlace(matrix, startCol, headDim, baseTheta, inverse);
        }
        private static void ApplyRotaryPositionEmbeddingHeadCoreInPlace(float[,] matrix, int startCol, int headDim, float baseTheta, bool inverse)
        {
            int seqLen = matrix.GetLength(0);
            int embeddingDim = matrix.GetLength(1);
            int pairCount = headDim / 2;

            if (seqLen == 0 || pairCount == 0)
            {
                return;
            }

            Span<float> data = FlatSpan(matrix);

            float[] rentedDenom = ArrayPool<float>.Shared.Rent(pairCount);

            try
            {
                Span<float> denom = rentedDenom.AsSpan(0, pairCount);
                for (int pair = 0; pair < pairCount; pair++)
                {
                    denom[pair] = MathF.Pow(baseTheta, (2f * pair) / headDim);
                }

                for (int pos = 0; pos < seqLen; pos++)
                {
                    int row = pos * embeddingDim;
                    for (int pair = 0; pair < pairCount; pair++)
                    {
                        int evenCol = startCol + (pair * 2);
                        int evenIndex = row + evenCol;
                        int oddIndex = evenIndex + 1;

                        float theta = pos / denom[pair];
                        float cos = MathF.Cos(theta);
                        float sin = MathF.Sin(theta);

                        if (inverse)
                        {
                            sin = -sin;
                        }

                        float x0 = data[evenIndex];
                        float x1 = data[oddIndex];

                        data[evenIndex] = (x0 * cos) - (x1 * sin);
                        data[oddIndex] = (x0 * sin) + (x1 * cos);
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rentedDenom);
            }
        }
        #endregion

        public void SigmoidInPlace(float[,] matrix)
        {
            Span<float> m = FlatSpan(matrix);

            for (int i = 0; i < m.Length; i++)
            {
                m[i] = StableSigmoidCore(m[i]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float StableSigmoid(float x)
        {
            return StableSigmoidCore(x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

        public void ZeroMatrixColumns(float[,] matrix, int columnCount)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            if (columnCount <= 0)
            {
                return;
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            int count = Math.Min(columnCount, cols);

            for (int i = 0; i < rows; i++)
            {
                Array.Clear(matrix, i * cols, count);
            }
        }
        public void VectorUpdateClamped(float[] weights, float[] gradients, float learningRate, float minValue, float maxValue)
        {
            if (weights == null)
            {
                throw new ArgumentNullException(nameof(weights));
            }
            if (gradients == null)
            {
                throw new ArgumentNullException(nameof(gradients));
            }
            if (weights.Length != gradients.Length)
            {
                throw new ArgumentException("weights and gradients must have the same length.");
            }

            for (int i = 0; i < weights.Length; i++)
            {
                float value = weights[i] - learningRate * gradients[i];
                weights[i] = Math.Clamp(value, minValue, maxValue);
            }
        }
        public void Matrix3DAddInPlace(float[,,] target, float[,,] addend)
        {
            if (target == null)
            {
                throw new ArgumentNullException(nameof(target));
            }
            if (addend == null)
            {
                throw new ArgumentNullException(nameof(addend));
            }

            int d0 = target.GetLength(0);
            int d1 = target.GetLength(1);
            int d2 = target.GetLength(2);

            if (addend.GetLength(0) != d0 || addend.GetLength(1) != d1 || addend.GetLength(2) != d2)
            {
                throw new ArgumentException("3D tensor shape mismatch.", nameof(addend));
            }

            Span<float> t = FlatSpan(target);
            ReadOnlySpan<float> a = FlatReadOnlySpan(addend);
            int i = 0;
            int end4 = t.Length & ~3;

            for (; i < end4; i += 4)
            {
                t[i] += a[i];
                t[i + 1] += a[i + 1];
                t[i + 2] += a[i + 2];
                t[i + 3] += a[i + 3];
            }

            for (; i < t.Length; i++)
            {
                t[i] += a[i];
            }
        }
        public void Matrix3DUpdate(float[,,] weights, float[,,] gradients, float learningRate)
        {
            if (weights == null)
            {
                throw new ArgumentNullException(nameof(weights));
            }
            if (gradients == null)
            {
                throw new ArgumentNullException(nameof(gradients));
            }

            int d0 = weights.GetLength(0);
            int d1 = weights.GetLength(1);
            int d2 = weights.GetLength(2);

            if (gradients.GetLength(0) != d0 || gradients.GetLength(1) != d1 || gradients.GetLength(2) != d2)
            {
                throw new ArgumentException("3D tensor shape mismatch.", nameof(gradients));
            }

            Span<float> w = FlatSpan(weights);
            ReadOnlySpan<float> g = FlatReadOnlySpan(gradients);

            int i = 0;
            int end4 = w.Length & ~3;

            for (; i < end4; i += 4)
            {
                w[i] -= learningRate * g[i];
                w[i + 1] -= learningRate * g[i + 1];
                w[i + 2] -= learningRate * g[i + 2];
                w[i + 3] -= learningRate * g[i + 3];
            }
            for (; i < w.Length; i++)
            {
                w[i] -= learningRate * g[i];
            }
        }

        public void Dispose()
        {

        }
    }
}