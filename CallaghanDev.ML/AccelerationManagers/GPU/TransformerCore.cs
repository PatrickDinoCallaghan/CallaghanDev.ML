using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    //Transformer Core
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private void InitTransformerCoreKernels()
        {
            //Softmax
            _softmaxKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>>(SoftmaxKernel);

            //LayerNorm
            _layerNormKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, float>(LayerNormKernel);

            ///LayerNormForward
            _layerNormForwardKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(LayerNormForwardKernel);

            //LayerNormBackward
            _layerNormBackwardKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(LayerNormBackwardKernel);

            // MultiHeadAttentionForward
            _multiHeadAttentionWeightsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, float, int>(MultiHeadAttentionWeightsKernel);
            _multiHeadAttentionOutputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(MultiHeadAttentionOutputKernel);

            //
        }

        #region Softmax

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<int, Stride2D.DenseX>> _softmaxKernel;

        public float[,] Softmax(float[,] matrix, bool[,] mask = null)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            var bufMask = mask != null ? _accelerator.Allocate2DDenseX<int>(new Index2D(rows, cols)) : _accelerator.Allocate2DDenseX<int>(new Index2D(1, 1));

            try
            {
                bufIn.CopyFromCPU(matrix);
                if (mask != null)
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

                _softmaxKernel(new Index1D(rows), bufIn.View, bufOut.View, bufMask.View);

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

        private static void SoftmaxKernel(Index1D row, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> output, ArrayView2D<int, Stride2D.DenseX> mask)
        {
            int cols = (int)input.Extent.Y;
            bool hasMask = mask.Extent.Size > 1;

            float max = float.NegativeInfinity;
            for (int j = 0; j < cols; j++)
            {

                if (!hasMask || mask[row, j] != 0)
                {
                    max = XMath.Max(max, input[row, j]);
                }
            }

            float sum = 0.0f;
            for (int j = 0; j < cols; j++)
            {
                if (hasMask && mask[row, j] == 0)
                    output[row, j] = 0.0f;
                else
                {
                    output[row, j] = XMath.Exp(input[row, j] - max);
                    sum += output[row, j];
                }
            }

            for (int j = 0; j < cols; j++)
            {
                output[row, j] /= sum;
            }
        }

        #endregion

        #region LayerNorm

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, float> _layerNormKernel;
        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

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

                _layerNormForwardKernel(new Index1D(batchSize),
                    bufIn.View, bufGamma.View, bufBeta.View,
                    bufOut.View, bufMeans.View, bufVariances.View,
                    epsilon);

                var output = new float[batchSize, features];
                var means = new float[batchSize];
                var variances = new float[batchSize];
                bufOut.CopyToCPU(output);
                bufMeans.CopyToCPU(means);
                bufVariances.CopyToCPU(variances);

                // Compute normalized on CPU from means/variances (avoids extra GPU buffer)
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
            mean = mean / features;
            means[batch] = mean;

            float variance = 0.0f;

            for (int j = 0; j < features; j++)
            {
                float diff = input[batch, j] - mean;
                variance += diff * diff;
            }

            variance = variance / features;
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

                // Zero out accumulation buffers
                bufDGamma.CopyFromCPU(new float[features]);
                bufDBeta.CopyFromCPU(new float[features]);
                bufDInput.CopyFromCPU(new float[batchSize, features]);

                _layerNormBackwardKernel(new Index1D(batchSize),
                    bufDOut.View, bufNormalized.View, bufGamma.View,
                    bufInput.View, bufMean.View, bufVariance.View,
                    bufDInput.View, bufDGamma.View, bufDBeta.View,
                    epsilon);

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
            return _mutliThreadCPU.CreateCausalMask(seqLen);
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

            if (numHeads <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numHeads));
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
            int attentionRows = numHeads * seqLenQ;

            var bufQ = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenQ, embeddingDim));
            var bufK = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenK, embeddingDim));
            var bufV = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenK, embeddingDim));
            var bufOutput = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenQ, embeddingDim));

            // Shape: [numHeads * seqLenQ, seqLenK]
            // Stores final attention weights.
            var bufWeights = _accelerator.Allocate2DDenseX<float>(new Index2D(attentionRows, seqLenK));

            int hasMask = mask != null ? 1 : 0;

            var bufMask = hasMask == 1 ? _accelerator.Allocate2DDenseX<int>(new Index2D(seqLenQ, seqLenK)) : _accelerator.Allocate2DDenseX<int>(new Index2D(1, 1));

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

                _multiHeadAttentionWeightsKernel(
                    new Index1D(attentionRows),
                    bufQ.View,
                    bufK.View,
                    bufMask.View,
                    bufWeights.View,
                    numHeads,
                    headDim,
                    scale,
                    hasMask);

                _multiHeadAttentionOutputKernel(
                    new Index1D(attentionRows * headDim),
                    bufWeights.View,
                    bufV.View,
                    bufOutput.View,
                    headDim);

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

            // First pass:
            // compute QK^T * scale for this head/query row
            // and find max for stable softmax.
            for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
            {
                if (hasMask != 0 && mask[queryIndex, keyIndex] == 0)
                {
                    weights[attentionRow, keyIndex] = float.NegativeInfinity;
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

            // Fully masked row.
            // Match your CPU version: output attention weights all zero.
            if (validCount == 0)
            {
                for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
                {
                    weights[attentionRow, keyIndex] = 0.0f;
                }

                return;
            }

            // Second pass:
            // stable softmax numerator.
            float sumExp = 0.0f;

            for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
            {
                float score = weights[attentionRow, keyIndex];

                if (float.IsNegativeInfinity(score))
                {
                    weights[attentionRow, keyIndex] = 0.0f;
                    continue;
                }

                float exp = XMath.Exp(score - maxScore);

                weights[attentionRow, keyIndex] = exp;
                sumExp += exp;
            }

            // Third pass:
            // normalize softmax.
            float invSumExp = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;

            for (int keyIndex = 0; keyIndex < seqLenK; keyIndex++)
            {
                weights[attentionRow, keyIndex] *= invSumExp;
            }
        }

        #endregion
    }
}
