using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private void InitTransformerCoreKernels()
        {
            _softmaxKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>, int>(SoftmaxKernel);

            _layerNormKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, float>(LayerNormKernel);

            _layerNormForwardKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(LayerNormForwardKernel);

            _layerNormBackwardKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(LayerNormBackwardKernel);

            _multiHeadAttentionWeightsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, float, int>(MultiHeadAttentionWeightsKernel);
            _multiHeadAttentionOutputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(MultiHeadAttentionOutputKernel);
        }

        private void DisposeTransformerCoreBuffers()
        {
        }

        #region Softmax

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>, int> _softmaxKernel;

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

            if (!ShouldUseGpu((long)rows * cols))
            {
                return _mutliThreadCPU.Softmax(matrix, mask);
            }

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufMask = mask != null
                ? _accelerator.Allocate2DDenseX<int>(new Index2D(rows, cols))
                : _accelerator.Allocate2DDenseX<int>(new Index2D(1, 1));

            try
            {
                bufIn.CopyFromCPU(matrix);

                int hasMask = mask != null ? 1 : 0;
                if (hasMask == 1)
                {
                    var intMask = new int[rows, cols];
                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < cols; j++)
                        {
                            intMask[i, j] = mask[i, j] ? 1 : 0;
                        }
                    }

                    bufMask.CopyFromCPU(intMask);
                }

                _softmaxKernel(new Index1D(rows), bufIn.View, bufOut.View, bufMask.View, hasMask);

                var result = new float[rows, cols];
                bufOut.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufIn.Dispose();
                bufOut.Dispose();
                bufMask.Dispose();
            }
        }

        private static void SoftmaxKernel(Index1D row, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> output, ArrayView2D<int, Stride2D.DenseX> mask, int hasMask)
        {
            int cols = (int)input.Extent.Y;

            float max = float.NegativeInfinity;
            int validCount = 0;

            for (int j = 0; j < cols; j++)
            {
                if (hasMask != 0 && mask[row, j] == 0)
                {
                    continue;
                }

                float value = input[row, j];
                if (value > max)
                {
                    max = value;
                }

                validCount++;
            }

            if (validCount == 0)
            {
                for (int j = 0; j < cols; j++)
                {
                    output[row, j] = 0.0f;
                }

                return;
            }

            float sum = 0.0f;
            for (int j = 0; j < cols; j++)
            {
                if (hasMask != 0 && mask[row, j] == 0)
                {
                    output[row, j] = 0.0f;
                    continue;
                }

                float exp = XMath.Exp(input[row, j] - max);
                output[row, j] = exp;
                sum += exp;
            }

            float invSum = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int j = 0; j < cols; j++)
            {
                output[row, j] *= invSum;
            }
        }

        #endregion

        #region LayerNorm

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, float> _layerNormKernel;

        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            if (!ShouldUseGpu((long)batchSize * features))
            {
                return _mutliThreadCPU.LayerNorm(input, gamma, beta, epsilon);
            }

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufGamma = _accelerator.Allocate1D<float>(features);
            var bufBeta = _accelerator.Allocate1D<float>(features);
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));

            try
            {
                bufIn.CopyFromCPU(input);
                bufGamma.CopyFromCPU(gamma);
                bufBeta.CopyFromCPU(beta);

                _layerNormKernel(new Index1D(batchSize), bufIn.View, bufGamma.View, bufBeta.View, bufOut.View, epsilon);

                var result = new float[batchSize, features];
                bufOut.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufIn.Dispose();
                bufGamma.Dispose();
                bufBeta.Dispose();
                bufOut.Dispose();
            }
        }

        private static void LayerNormKernel(Index1D batch, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> beta, ArrayView2D<float, Stride2D.DenseX> output, float epsilon)
        {
            int features = (int)input.Extent.Y;

            float mean = 0.0f;
            for (int j = 0; j < features; j++)
            {
                mean += input[batch, j];
            }
            mean /= features;

            float variance = 0.0f;
            for (int j = 0; j < features; j++)
            {
                float diff = input[batch, j] - mean;
                variance += diff * diff;
            }
            variance /= features;

            float stdDev = XMath.Sqrt(variance + epsilon);
            for (int j = 0; j < features; j++)
            {
                output[batch, j] = gamma[j] * (input[batch, j] - mean) / stdDev + beta[j];
            }
        }

        #endregion

        #region LayerNormForward

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _layerNormForwardKernel;

        public (float[,] output, float[] means, float[] variances, float[,] normalized) LayerNormForward(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            if (!ShouldUseGpu((long)batchSize * features))
            {
                return _mutliThreadCPU.LayerNormForward(input, gamma, beta, epsilon);
            }

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufGamma = _accelerator.Allocate1D<float>(features);
            var bufBeta = _accelerator.Allocate1D<float>(features);
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufMeans = _accelerator.Allocate1D<float>(batchSize);
            var bufVariances = _accelerator.Allocate1D<float>(batchSize);

            try
            {
                bufIn.CopyFromCPU(input);
                bufGamma.CopyFromCPU(gamma);
                bufBeta.CopyFromCPU(beta);

                _layerNormForwardKernel(new Index1D(batchSize), bufIn.View, bufGamma.View, bufBeta.View, bufOut.View, bufMeans.View, bufVariances.View, epsilon);

                var output = new float[batchSize, features];
                var means = new float[batchSize];
                var variances = new float[batchSize];
                bufOut.CopyToCPU(output);
                bufMeans.CopyToCPU(means);
                bufVariances.CopyToCPU(variances);

                var normalized = new float[batchSize, features];
                for (int i = 0; i < batchSize; i++)
                {
                    float stdDev = MathF.Sqrt(variances[i] + epsilon);
                    for (int j = 0; j < features; j++)
                    {
                        normalized[i, j] = (input[i, j] - means[i]) / stdDev;
                    }
                }

                return (output, means, variances, normalized);
            }
            finally
            {
                bufIn.Dispose();
                bufGamma.Dispose();
                bufBeta.Dispose();
                bufOut.Dispose();
                bufMeans.Dispose();
                bufVariances.Dispose();
            }
        }

        private static void LayerNormForwardKernel(Index1D batch, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> beta, ArrayView2D<float, Stride2D.DenseX> output, ArrayView1D<float, Stride1D.Dense> means, ArrayView1D<float, Stride1D.Dense> variances, float epsilon)
        {
            int features = (int)input.Extent.Y;

            float mean = 0.0f;
            for (int j = 0; j < features; j++)
            {
                mean += input[batch, j];
            }
            mean /= features;
            means[batch] = mean;

            float variance = 0.0f;
            for (int j = 0; j < features; j++)
            {
                float diff = input[batch, j] - mean;
                variance += diff * diff;
            }
            variance /= features;
            variances[batch] = variance;

            float stdDev = XMath.Sqrt(variance + epsilon);
            for (int j = 0; j < features; j++)
            {
                output[batch, j] = gamma[j] * (input[batch, j] - mean) / stdDev + beta[j];
            }
        }

        #endregion

        #region LayerNormBackward

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _layerNormBackwardKernel;

        public (float[,] dInput, float[] dGamma, float[] dBeta) LayerNormBackward(float[,] dOut, float[,] normalized, float[] gamma, float[,] input, float[] mean, float[] variance, float epsilon = 1e-5f)
        {
            int batchSize = dOut.GetLength(0);
            int features = dOut.GetLength(1);

            if (!ShouldUseGpu((long)batchSize * features))
            {
                return _mutliThreadCPU.LayerNormBackward(dOut, normalized, gamma, input, mean, variance, epsilon);
            }

            var bufDOut = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufNormalized = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufGamma = _accelerator.Allocate1D<float>(features);
            var bufInput = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufMean = _accelerator.Allocate1D<float>(batchSize);
            var bufVariance = _accelerator.Allocate1D<float>(batchSize);
            var bufDInput = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufDGamma = _accelerator.Allocate1D<float>(features);
            var bufDBeta = _accelerator.Allocate1D<float>(features);

            try
            {
                bufDOut.CopyFromCPU(dOut);
                bufNormalized.CopyFromCPU(normalized);
                bufGamma.CopyFromCPU(gamma);
                bufInput.CopyFromCPU(input);
                bufMean.CopyFromCPU(mean);
                bufVariance.CopyFromCPU(variance);
                bufDInput.MemSetToZero();
                bufDGamma.MemSetToZero();
                bufDBeta.MemSetToZero();

                _layerNormBackwardKernel(new Index1D(batchSize), bufDOut.View, bufNormalized.View, bufGamma.View, bufInput.View, bufMean.View, bufVariance.View, bufDInput.View, bufDGamma.View, bufDBeta.View, epsilon);

                var dInput = new float[batchSize, features];
                var dGamma = new float[features];
                var dBeta = new float[features];
                bufDInput.CopyToCPU(dInput);
                bufDGamma.CopyToCPU(dGamma);
                bufDBeta.CopyToCPU(dBeta);

                return (dInput, dGamma, dBeta);
            }
            finally
            {
                bufDOut.Dispose();
                bufNormalized.Dispose();
                bufGamma.Dispose();
                bufInput.Dispose();
                bufMean.Dispose();
                bufVariance.Dispose();
                bufDInput.Dispose();
                bufDGamma.Dispose();
                bufDBeta.Dispose();
            }
        }

        private static void LayerNormBackwardKernel(Index1D batch, ArrayView2D<float, Stride2D.DenseX> dOut, ArrayView2D<float, Stride2D.DenseX> normalized, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> mean, ArrayView1D<float, Stride1D.Dense> variance, ArrayView2D<float, Stride2D.DenseX> dInput, ArrayView1D<float, Stride1D.Dense> dGamma, ArrayView1D<float, Stride1D.Dense> dBeta, float epsilon)
        {
            int features = (int)input.Extent.Y;
            float invStd = 1.0f / XMath.Sqrt(variance[batch] + epsilon);

            for (int j = 0; j < features; j++)
            {
                Atomic.Add(ref dGamma[j], dOut[batch, j] * normalized[batch, j]);
                Atomic.Add(ref dBeta[j], dOut[batch, j]);
            }

            float dVar = 0.0f;
            float dMean = 0.0f;
            float invStdCubed = invStd * invStd * invStd;

            for (int j = 0; j < features; j++)
            {
                float dNorm = dOut[batch, j] * gamma[j];
                float xMinusMean = input[batch, j] - mean[batch];
                dVar += dNorm * xMinusMean * (-0.5f) * invStdCubed;
                dMean += dNorm * (-invStd);
            }

            float invN = 1.0f / features;
            for (int j = 0; j < features; j++)
            {
                float dNorm = dOut[batch, j] * gamma[j];
                float xMinusMean = input[batch, j] - mean[batch];
                dInput[batch, j] = dNorm * invStd + dVar * 2.0f * xMinusMean * invN + dMean * invN;
            }
        }

        #endregion

        #region CreateCausalMask

        public bool[,] CreateCausalMask(int seqLen)
        {
            if (seqLen < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(seqLen));
            }

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

        #endregion

        #region MultiHeadAttentionForward

        [Obsolete]
        public float[,] MultiHeadAttentionForward_Obsolete(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads");
            }

            int headDim = embeddingDim / numHeads;
            var concatenated = new float[seqLenQ, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                var Q_head = new float[seqLenQ, headDim];
                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        Q_head[i, j] = Q[i, startIdx + j];
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

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, float, int> _multiHeadAttentionWeightsKernel;
        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int> _multiHeadAttentionOutputKernel;

        public float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            if (Q == null) throw new ArgumentNullException(nameof(Q));
            if (K == null) throw new ArgumentNullException(nameof(K));
            if (V == null) throw new ArgumentNullException(nameof(V));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));

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
            int attentionRows = numHeads * seqLenQ;

            if (!ShouldUseGpu((long)numHeads * seqLenQ * seqLenK * headDim, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.MultiHeadAttentionForward(Q, K, V, numHeads, scale, mask);
            }

            var bufQ = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenQ, embeddingDim));
            var bufK = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenK, embeddingDim));
            var bufV = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenK, embeddingDim));
            var bufOutput = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenQ, embeddingDim));
            var bufWeights = _accelerator.Allocate2DDenseX<float>(new Index2D(attentionRows, seqLenK));

            int hasMask = mask != null ? 1 : 0;
            var bufMask = hasMask == 1
                ? _accelerator.Allocate2DDenseX<int>(new Index2D(seqLenQ, seqLenK))
                : _accelerator.Allocate2DDenseX<int>(new Index2D(1, 1));

            try
            {
                bufQ.CopyFromCPU(Q);
                bufK.CopyFromCPU(K);
                bufV.CopyFromCPU(V);

                if (hasMask == 1)
                {
                    var intMask = new int[seqLenQ, seqLenK];
                    for (int i = 0; i < seqLenQ; i++)
                    {
                        for (int j = 0; j < seqLenK; j++)
                        {
                            intMask[i, j] = mask[i, j] ? 1 : 0;
                        }
                    }

                    bufMask.CopyFromCPU(intMask);
                }

                _multiHeadAttentionWeightsKernel(new Index1D(attentionRows), bufQ.View, bufK.View, bufMask.View, bufWeights.View, numHeads, headDim, scale, hasMask);
                _multiHeadAttentionOutputKernel(new Index1D(attentionRows * headDim), bufWeights.View, bufV.View, bufOutput.View, headDim);

                var result = new float[seqLenQ, embeddingDim];
                bufOutput.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufQ.Dispose();
                bufK.Dispose();
                bufV.Dispose();
                bufOutput.Dispose();
                bufWeights.Dispose();
                bufMask.Dispose();
            }
        }

        private static void MultiHeadAttentionOutputKernel(Index1D index, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> V, ArrayView2D<float, Stride2D.DenseX> output, int headDim)
        {
            int seqLenQ = (int)output.Extent.X;
            int seqLenK = (int)V.Extent.X;

            int flat = index;
            int d = flat % headDim;
            int attentionRow = flat / headDim;
            int head = attentionRow / seqLenQ;
            int queryIndex = attentionRow - head * seqLenQ;
            int offset = head * headDim;

            float value = 0.0f;
            for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
            {
                value += weights[attentionRow, keyIndex] * V[keyIndex, offset + d];
            }

            output[queryIndex, offset + d] = value;
        }

        private static void MultiHeadAttentionWeightsKernel(Index1D index, ArrayView2D<float, Stride2D.DenseX> Q, ArrayView2D<float, Stride2D.DenseX> K, ArrayView2D<int, Stride2D.DenseX> mask, ArrayView2D<float, Stride2D.DenseX> weights, int numHeads, int headDim, float scale, int hasMask)
        {
            int attentionRow = index;
            int seqLenQ = (int)Q.Extent.X;
            int seqLenK = (int)K.Extent.X;
            int head = attentionRow / seqLenQ;
            int queryIndex = attentionRow - head * seqLenQ;
            int offset = head * headDim;

            float maxScore = float.NegativeInfinity;
            int validCount = 0;

            for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
            {
                if (hasMask != 0 && mask[queryIndex, keyIndex] == 0)
                {
                    weights[attentionRow, keyIndex] = 0.0f;
                    continue;
                }

                float dot = 0.0f;
                for (int d = 0; d < headDim; d++)
                {
                    dot += Q[queryIndex, offset + d] * K[keyIndex, offset + d];
                }

                float score = dot * scale;
                weights[attentionRow, keyIndex] = score;

                if (score > maxScore)
                {
                    maxScore = score;
                }

                validCount++;
            }

            if (validCount == 0)
            {
                for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
                {
                    weights[attentionRow, keyIndex] = 0.0f;
                }

                return;
            }

            float sumExp = 0.0f;
            for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
            {
                if (hasMask != 0 && mask[queryIndex, keyIndex] == 0)
                {
                    weights[attentionRow, keyIndex] = 0.0f;
                    continue;
                }

                float exp = XMath.Exp(weights[attentionRow, keyIndex] - maxScore);
                weights[attentionRow, keyIndex] = exp;
                sumExp += exp;
            }

            float invSumExp = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
            for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
            {
                weights[attentionRow, keyIndex] *= invSumExp;
            }
        }

        #endregion

        #region MultiHeadAttentionBackward

        [Obsolete]
        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward_Obsolete(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            return MultiHeadAttentionBackward(Q, K, V, dConcatenated, numHeads, scale, useDecoderMask);
        }

        [Obsolete]
        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward_Obsolete(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask)
        {
            return MultiHeadAttentionBackward(Q, K, V, dConcatenated, numHeads, scale, mask);
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask)
        {
            return MultiHeadAttentionBackwardCore(Q, K, V, dConcatenated, numHeads, scale, mask, useDecoderMask: false);
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            return MultiHeadAttentionBackwardCore(Q, K, V, dConcatenated, numHeads, scale, mask: null, useDecoderMask: useDecoderMask);
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
            {
                throw new ArgumentException("Q, K and V must have the same embedding dimension.");
            }

            if (V.GetLength(0) != seqLenK)
            {
                throw new ArgumentException("K and V must have the same sequence length.");
            }

            if (dConcatenated.GetLength(0) != seqLenQ || dConcatenated.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("dConcatenated shape must match Q shape.", nameof(dConcatenated));
            }

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads.", nameof(numHeads));
            }

            if (mask != null && (mask.GetLength(0) != seqLenQ || mask.GetLength(1) != seqLenK))
            {
                throw new ArgumentException($"Mask shape must be [{seqLenQ},{seqLenK}], got [{mask.GetLength(0)},{mask.GetLength(1)}].", nameof(mask));
            }

            int headDim = embeddingDim / numHeads;
            var dQFull = new float[seqLenQ, embeddingDim];
            var dKFull = new float[seqLenK, embeddingDim];
            var dVFull = new float[seqLenK, embeddingDim];

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
                    Array.Clear(weights, 0, weights.Length);
                    Array.Clear(dAttn, 0, dAttn.Length);
                    Array.Clear(dqRow, 0, dqRow.Length);

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
                            {
                                continue;
                            }

                            float dot = 0.0f;
                            for (int k = 0; k < headDim; k++)
                            {
                                dot += qRow[k] * K[j, offset + k];
                            }

                            float score = dot * scale;
                            weights[j] = score;
                            activeIndices[activeCount++] = j;

                            if (score > maxScore)
                            {
                                maxScore = score;
                            }
                        }
                    }
                    else
                    {
                        int visibleKeys = useDecoderMask ? Math.Min(i + 1, seqLenK) : seqLenK;
                        activeCount = visibleKeys;

                        for (int j = 0; j < visibleKeys; j++)
                        {
                            float dot = 0.0f;
                            for (int k = 0; k < headDim; k++)
                            {
                                dot += qRow[k] * K[j, offset + k];
                            }

                            float score = dot * scale;
                            weights[j] = score;

                            if (score > maxScore)
                            {
                                maxScore = score;
                            }
                        }
                    }

                    if (activeCount == 0 || float.IsNegativeInfinity(maxScore))
                    {
                        continue;
                    }

                    float sumExp = 0.0f;

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

                    if (sumExp <= 0.0f)
                    {
                        continue;
                    }

                    float invSumExp = 1.0f / sumExp;

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
                        {
                            weights[j] *= invSumExp;
                        }
                    }

                    float rowDot = 0.0f;

                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];
                            float w = weights[j];
                            float dAttnJ = 0.0f;

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
                            float dAttnJ = 0.0f;

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

                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];
                            float dDot = weights[j] * (dAttn[j] - rowDot) * scale;

                            if (dDot == 0.0f)
                            {
                                continue;
                            }

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

                            if (dDot == 0.0f)
                            {
                                continue;
                            }

                            for (int k = 0; k < headDim; k++)
                            {
                                int col = offset + k;
                                dqRow[k] += dDot * K[j, col];
                                dKFull[j, col] += dDot * qRow[k];
                            }
                        }
                    }

                    for (int k = 0; k < headDim; k++)
                    {
                        dQFull[i, offset + k] += dqRow[k];
                    }
                }
            }

            return (dQFull, dKFull, dVFull);
        }

        #endregion
    }
}
