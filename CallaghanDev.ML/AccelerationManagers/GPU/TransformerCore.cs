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

            _scaledDotProductAttentionWeightsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, float, int, int>(ScaledDotProductAttentionWeightsKernel);


            _projectQKVKernel =
    _accelerator.LoadAutoGroupedStreamKernel<
        Index1D,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView2D<float, Stride2D.DenseX>>(
            ProjectQKVKernel);

            _backpropQKVDInputKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>>(
                BackpropQKVDInputKernel);

            _backpropQKVWeightGradKernel =
                _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>>(
                        BackpropQKVWeightGradKernel);

            _backpropQKVBiasGradKernel =
                _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>>(
                        BackpropQKVBiasGradKernel);

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

        #region ScaledDotProductAttention

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, float, int, int> _scaledDotProductAttentionWeightsKernel;
        private static void ScaledDotProductAttentionWeightsKernel(Index1D index, ArrayView2D<float, Stride2D.DenseX> Q, ArrayView2D<float, Stride2D.DenseX> K, ArrayView2D<int, Stride2D.DenseX> mask, ArrayView2D<float, Stride2D.DenseX> weights, int numHeads, int headDim, float scale, int hasMask, int causal)
        {
            int attentionRow = index;
            int seqLenQ = (int)Q.Extent.X;
            int seqLenK = (int)K.Extent.X;

            int head = attentionRow / seqLenQ;
            int queryIndex = attentionRow - head * seqLenQ;
            int offset = head * headDim;

            int usableKeyLen = seqLenK;

            if (causal != 0 && queryIndex + 1 < seqLenK)
            {
                usableKeyLen = queryIndex + 1;
            }

            float maxScore = float.NegativeInfinity;
            int validCount = 0;

            for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
            {
                if (keyIndex >= usableKeyLen || (hasMask != 0 && mask[queryIndex, keyIndex] == 0))
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
                if (keyIndex >= usableKeyLen || (hasMask != 0 && mask[queryIndex, keyIndex] == 0))
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
            int attentionRows = numHeads * queryLen;

            long workUnits = (long)numHeads * queryLen * keyLen * headDim;

            if (!ShouldUseGpu(workUnits, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.ScaledDotProductAttention(q, k,  v, numHeads, mask, causal);
            }

            var bufQ = _accelerator.Allocate2DDenseX<float>(new Index2D(queryLen, embeddingDim));
            var bufK = _accelerator.Allocate2DDenseX<float>(new Index2D(keyLen, embeddingDim));
            var bufV = _accelerator.Allocate2DDenseX<float>(new Index2D(keyLen, embeddingDim));
            var bufOutput = _accelerator.Allocate2DDenseX<float>(new Index2D(queryLen, embeddingDim));
            var bufWeights = _accelerator.Allocate2DDenseX<float>(new Index2D(attentionRows, keyLen));

            int hasMask = mask != null ? 1 : 0;
            int useCausal = causal ? 1 : 0;

            var bufMask = hasMask == 1 ? _accelerator.Allocate2DDenseX<int>(new Index2D(queryLen, keyLen)) : _accelerator.Allocate2DDenseX<int>(new Index2D(1, 1));

            try
            {
                bufQ.CopyFromCPU(q);
                bufK.CopyFromCPU(k);
                bufV.CopyFromCPU(v);

                if (hasMask == 1)
                {
                    var intMask = new int[queryLen, keyLen];

                    for (int i = 0; i < queryLen; i++)
                    {
                        for (int j = 0; j < keyLen; j++)
                        {
                            intMask[i, j] = mask[i, j] ? 1 : 0;
                        }
                    }

                    bufMask.CopyFromCPU(intMask);
                }

                _scaledDotProductAttentionWeightsKernel(new Index1D(attentionRows), bufQ.View, bufK.View, bufMask.View, bufWeights.View, numHeads, headDim, scale, hasMask, useCausal);

                _multiHeadAttentionOutputKernel(new Index1D(attentionRows * headDim),bufWeights.View, bufV.View, bufOutput.View, headDim);

                var result = new float[queryLen, embeddingDim];
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

        #endregion

        #region Fused QKV Projection

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _projectQKVKernel;

        private Action<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> _backpropQKVDInputKernel;

        private Action<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> _backpropQKVWeightGradKernel;

        private Action<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> _backpropQKVBiasGradKernel;

        public (float[,] Q, float[,] K, float[,] V) ProjectQKV(float[,] input, float[,] WQ, float[] biasQ, float[,] WK, float[] biasK, float[,] WV, float[] biasV)
        {
            var shape = ValidateProjectQKVInputsGpu(
                input,
                WQ,
                biasQ,
                WK,
                biasK,
                WV,
                biasV);

            int rows = shape.Rows;
            int inputDim = shape.InputDim;
            int outputDim = shape.OutputDim;

            long workUnits = 3L * rows * outputDim * inputDim;

            if (!ShouldUseGpu(workUnits, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.ProjectQKV(
                    input,
                    WQ,
                    biasQ,
                    WK,
                    biasK,
                    WV,
                    biasV);
            }

            var bufInput = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, inputDim));

            var bufWQ = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim));
            var bufWK = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim));
            var bufWV = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim));

            var bufBiasQ = _accelerator.Allocate1D<float>(outputDim);
            var bufBiasK = _accelerator.Allocate1D<float>(outputDim);
            var bufBiasV = _accelerator.Allocate1D<float>(outputDim);

            var bufQ = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, outputDim));
            var bufK = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, outputDim));
            var bufV = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, outputDim));

            try
            {
                bufInput.CopyFromCPU(input);

                bufWQ.CopyFromCPU(WQ);
                bufWK.CopyFromCPU(WK);
                bufWV.CopyFromCPU(WV);

                bufBiasQ.CopyFromCPU(biasQ);
                bufBiasK.CopyFromCPU(biasK);
                bufBiasV.CopyFromCPU(biasV);

                _projectQKVKernel(
                    new Index1D(rows * outputDim),
                    bufInput.View,
                    bufWQ.View,
                    bufBiasQ.View,
                    bufWK.View,
                    bufBiasK.View,
                    bufWV.View,
                    bufBiasV.View,
                    bufQ.View,
                    bufK.View,
                    bufV.View);

                var Q = new float[rows, outputDim];
                var K = new float[rows, outputDim];
                var V = new float[rows, outputDim];

                bufQ.CopyToCPU(Q);
                bufK.CopyToCPU(K);
                bufV.CopyToCPU(V);

                return (Q, K, V);
            }
            finally
            {
                bufInput.Dispose();

                bufWQ.Dispose();
                bufWK.Dispose();
                bufWV.Dispose();

                bufBiasQ.Dispose();
                bufBiasK.Dispose();
                bufBiasV.Dispose();

                bufQ.Dispose();
                bufK.Dispose();
                bufV.Dispose();
            }
        }

        public float[,] BackpropQKV(
            float[,] input,
            float[,] dQ,
            float[,] dK,
            float[,] dV,
            float[,] WQ,
            float[,] WK,
            float[,] WV,
            float[,] WQGrad,
            float[] biasQGrad,
            float[,] WKGrad,
            float[] biasKGrad,
            float[,] WVGrad,
            float[] biasVGrad)
        {
            var shape = ValidateBackpropQKVInputsGpu(
                input,
                dQ,
                dK,
                dV,
                WQ,
                WK,
                WV,
                WQGrad,
                biasQGrad,
                WKGrad,
                biasKGrad,
                WVGrad,
                biasVGrad);

            int rows = shape.Rows;
            int inputDim = shape.InputDim;
            int outputDim = shape.OutputDim;

            long workUnits = 6L * rows * inputDim * outputDim;

            if (!ShouldUseGpu(workUnits, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.BackpropQKV(
                    input,
                    dQ,
                    dK,
                    dV,
                    WQ,
                    WK,
                    WV,
                    WQGrad,
                    biasQGrad,
                    WKGrad,
                    biasKGrad,
                    WVGrad,
                    biasVGrad);
            }

            var bufInput = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, inputDim));

            var bufDQ = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, outputDim));
            var bufDK = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, outputDim));
            var bufDV = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, outputDim));

            var bufWQ = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim));
            var bufWK = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim));
            var bufWV = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim));

            var bufWQGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim));
            var bufWKGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim));
            var bufWVGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim));

            var bufBiasQGrad = _accelerator.Allocate1D<float>(outputDim);
            var bufBiasKGrad = _accelerator.Allocate1D<float>(outputDim);
            var bufBiasVGrad = _accelerator.Allocate1D<float>(outputDim);

            var bufDInput = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, inputDim));

            try
            {
                bufInput.CopyFromCPU(input);

                bufDQ.CopyFromCPU(dQ);
                bufDK.CopyFromCPU(dK);
                bufDV.CopyFromCPU(dV);

                bufWQ.CopyFromCPU(WQ);
                bufWK.CopyFromCPU(WK);
                bufWV.CopyFromCPU(WV);

                // Important: these may already contain accumulated gradients.
                // Copy them in, then kernels add to them.
                bufWQGrad.CopyFromCPU(WQGrad);
                bufWKGrad.CopyFromCPU(WKGrad);
                bufWVGrad.CopyFromCPU(WVGrad);

                bufBiasQGrad.CopyFromCPU(biasQGrad);
                bufBiasKGrad.CopyFromCPU(biasKGrad);
                bufBiasVGrad.CopyFromCPU(biasVGrad);

                _backpropQKVDInputKernel(
                    new Index1D(rows * inputDim),
                    bufDQ.View,
                    bufDK.View,
                    bufDV.View,
                    bufWQ.View,
                    bufWK.View,
                    bufWV.View,
                    bufDInput.View);

                _backpropQKVWeightGradKernel(
                    new Index1D(outputDim * inputDim),
                    bufInput.View,
                    bufDQ.View,
                    bufDK.View,
                    bufDV.View,
                    bufWQGrad.View,
                    bufWKGrad.View,
                    bufWVGrad.View);

                _backpropQKVBiasGradKernel(
                    new Index1D(outputDim),
                    bufDQ.View,
                    bufDK.View,
                    bufDV.View,
                    bufBiasQGrad.View,
                    bufBiasKGrad.View,
                    bufBiasVGrad.View);

                var dInput = new float[rows, inputDim];

                bufDInput.CopyToCPU(dInput);

                bufWQGrad.CopyToCPU(WQGrad);
                bufWKGrad.CopyToCPU(WKGrad);
                bufWVGrad.CopyToCPU(WVGrad);

                bufBiasQGrad.CopyToCPU(biasQGrad);
                bufBiasKGrad.CopyToCPU(biasKGrad);
                bufBiasVGrad.CopyToCPU(biasVGrad);

                return dInput;
            }
            finally
            {
                bufInput.Dispose();

                bufDQ.Dispose();
                bufDK.Dispose();
                bufDV.Dispose();

                bufWQ.Dispose();
                bufWK.Dispose();
                bufWV.Dispose();

                bufWQGrad.Dispose();
                bufWKGrad.Dispose();
                bufWVGrad.Dispose();

                bufBiasQGrad.Dispose();
                bufBiasKGrad.Dispose();
                bufBiasVGrad.Dispose();

                bufDInput.Dispose();
            }
        }

        private static void ProjectQKVKernel(
            Index1D index,
            ArrayView2D<float, Stride2D.DenseX> input,
            ArrayView2D<float, Stride2D.DenseX> WQ,
            ArrayView1D<float, Stride1D.Dense> biasQ,
            ArrayView2D<float, Stride2D.DenseX> WK,
            ArrayView1D<float, Stride1D.Dense> biasK,
            ArrayView2D<float, Stride2D.DenseX> WV,
            ArrayView1D<float, Stride1D.Dense> biasV,
            ArrayView2D<float, Stride2D.DenseX> Q,
            ArrayView2D<float, Stride2D.DenseX> K,
            ArrayView2D<float, Stride2D.DenseX> V)
        {
            int outputDim = (int)Q.Extent.Y;
            int inputDim = (int)input.Extent.Y;

            int flat = index;
            int row = flat / outputDim;
            int o = flat - row * outputDim;

            float qSum = biasQ[o];
            float kSum = biasK[o];
            float vSum = biasV[o];

            for (int d = 0; d < inputDim; d++)
            {
                float x = input[row, d];

                qSum += WQ[o, d] * x;
                kSum += WK[o, d] * x;
                vSum += WV[o, d] * x;
            }

            Q[row, o] = qSum;
            K[row, o] = kSum;
            V[row, o] = vSum;
        }

        private static void BackpropQKVDInputKernel(
            Index1D index,
            ArrayView2D<float, Stride2D.DenseX> dQ,
            ArrayView2D<float, Stride2D.DenseX> dK,
            ArrayView2D<float, Stride2D.DenseX> dV,
            ArrayView2D<float, Stride2D.DenseX> WQ,
            ArrayView2D<float, Stride2D.DenseX> WK,
            ArrayView2D<float, Stride2D.DenseX> WV,
            ArrayView2D<float, Stride2D.DenseX> dInput)
        {
            int inputDim = (int)dInput.Extent.Y;
            int outputDim = (int)dQ.Extent.Y;

            int flat = index;
            int row = flat / inputDim;
            int d = flat - row * inputDim;

            float sum = 0.0f;

            for (int o = 0; o < outputDim; o++)
            {
                sum += dQ[row, o] * WQ[o, d]
                     + dK[row, o] * WK[o, d]
                     + dV[row, o] * WV[o, d];
            }

            dInput[row, d] = sum;
        }

        private static void BackpropQKVWeightGradKernel(
            Index1D index,
            ArrayView2D<float, Stride2D.DenseX> input,
            ArrayView2D<float, Stride2D.DenseX> dQ,
            ArrayView2D<float, Stride2D.DenseX> dK,
            ArrayView2D<float, Stride2D.DenseX> dV,
            ArrayView2D<float, Stride2D.DenseX> WQGrad,
            ArrayView2D<float, Stride2D.DenseX> WKGrad,
            ArrayView2D<float, Stride2D.DenseX> WVGrad)
        {
            int rows = (int)input.Extent.X;
            int inputDim = (int)input.Extent.Y;

            int flat = index;
            int o = flat / inputDim;
            int d = flat - o * inputDim;

            float qSum = 0.0f;
            float kSum = 0.0f;
            float vSum = 0.0f;

            for (int row = 0; row < rows; row++)
            {
                float x = input[row, d];

                qSum += dQ[row, o] * x;
                kSum += dK[row, o] * x;
                vSum += dV[row, o] * x;
            }

            WQGrad[o, d] += qSum;
            WKGrad[o, d] += kSum;
            WVGrad[o, d] += vSum;
        }

        private static void BackpropQKVBiasGradKernel(
            Index1D index,
            ArrayView2D<float, Stride2D.DenseX> dQ,
            ArrayView2D<float, Stride2D.DenseX> dK,
            ArrayView2D<float, Stride2D.DenseX> dV,
            ArrayView1D<float, Stride1D.Dense> biasQGrad,
            ArrayView1D<float, Stride1D.Dense> biasKGrad,
            ArrayView1D<float, Stride1D.Dense> biasVGrad)
        {
            int rows = (int)dQ.Extent.X;
            int o = index;

            float qSum = 0.0f;
            float kSum = 0.0f;
            float vSum = 0.0f;

            for (int row = 0; row < rows; row++)
            {
                qSum += dQ[row, o];
                kSum += dK[row, o];
                vSum += dV[row, o];
            }

            biasQGrad[o] += qSum;
            biasKGrad[o] += kSum;
            biasVGrad[o] += vSum;
        }

        private static (int Rows, int InputDim, int OutputDim) ValidateProjectQKVInputsGpu(
            float[,] input,
            float[,] WQ,
            float[] biasQ,
            float[,] WK,
            float[] biasK,
            float[,] WV,
            float[] biasV)
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

            if (rows <= 0)
                throw new ArgumentException("Input must contain at least one row.", nameof(input));

            if (inputDim <= 0)
                throw new ArgumentException("Input must contain at least one column.", nameof(input));

            int qOut = WQ.GetLength(0);
            int kOut = WK.GetLength(0);
            int vOut = WV.GetLength(0);

            if (qOut <= 0)
                throw new ArgumentException("WQ must contain at least one output row.", nameof(WQ));

            if (qOut != kOut || qOut != vOut)
                throw new ArgumentException("WQ, WK and WV output dimensions must match.");

            if (WQ.GetLength(1) != inputDim)
                throw new ArgumentException("WQ input dimension must match input width.", nameof(WQ));

            if (WK.GetLength(1) != inputDim)
                throw new ArgumentException("WK input dimension must match input width.", nameof(WK));

            if (WV.GetLength(1) != inputDim)
                throw new ArgumentException("WV input dimension must match input width.", nameof(WV));

            if (biasQ.Length != qOut)
                throw new ArgumentException("biasQ length must match WQ output dimension.", nameof(biasQ));

            if (biasK.Length != qOut)
                throw new ArgumentException("biasK length must match WK output dimension.", nameof(biasK));

            if (biasV.Length != qOut)
                throw new ArgumentException("biasV length must match WV output dimension.", nameof(biasV));

            return (rows, inputDim, qOut);
        }

        private static (int Rows, int InputDim, int OutputDim) ValidateBackpropQKVInputsGpu(
            float[,] input,
            float[,] dQ,
            float[,] dK,
            float[,] dV,
            float[,] WQ,
            float[,] WK,
            float[,] WV,
            float[,] WQGrad,
            float[] biasQGrad,
            float[,] WKGrad,
            float[] biasKGrad,
            float[,] WVGrad,
            float[] biasVGrad)
        {
            if (dQ == null) throw new ArgumentNullException(nameof(dQ));
            if (dK == null) throw new ArgumentNullException(nameof(dK));
            if (dV == null) throw new ArgumentNullException(nameof(dV));
            if (WQGrad == null) throw new ArgumentNullException(nameof(WQGrad));
            if (WKGrad == null) throw new ArgumentNullException(nameof(WKGrad));
            if (WVGrad == null) throw new ArgumentNullException(nameof(WVGrad));
            if (biasQGrad == null) throw new ArgumentNullException(nameof(biasQGrad));
            if (biasKGrad == null) throw new ArgumentNullException(nameof(biasKGrad));
            if (biasVGrad == null) throw new ArgumentNullException(nameof(biasVGrad));

            var shape = ValidateProjectQKVInputsGpu(
                input,
                WQ,
                biasQGrad,
                WK,
                biasKGrad,
                WV,
                biasVGrad);

            int rows = shape.Rows;
            int inputDim = shape.InputDim;
            int outputDim = shape.OutputDim;

            if (dQ.GetLength(0) != rows || dQ.GetLength(1) != outputDim)
                throw new ArgumentException($"dQ shape must be [{rows},{outputDim}].", nameof(dQ));

            if (dK.GetLength(0) != rows || dK.GetLength(1) != outputDim)
                throw new ArgumentException($"dK shape must be [{rows},{outputDim}].", nameof(dK));

            if (dV.GetLength(0) != rows || dV.GetLength(1) != outputDim)
                throw new ArgumentException($"dV shape must be [{rows},{outputDim}].", nameof(dV));

            if (WQGrad.GetLength(0) != outputDim || WQGrad.GetLength(1) != inputDim)
                throw new ArgumentException($"WQGrad shape must be [{outputDim},{inputDim}].", nameof(WQGrad));

            if (WKGrad.GetLength(0) != outputDim || WKGrad.GetLength(1) != inputDim)
                throw new ArgumentException($"WKGrad shape must be [{outputDim},{inputDim}].", nameof(WKGrad));

            if (WVGrad.GetLength(0) != outputDim || WVGrad.GetLength(1) != inputDim)
                throw new ArgumentException($"WVGrad shape must be [{outputDim},{inputDim}].", nameof(WVGrad));

            return shape;
        }

        #endregion
    }

}
