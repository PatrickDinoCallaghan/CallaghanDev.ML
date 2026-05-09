using CallaghanDev.ML.Transformers.TACAMT;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>> _applyContextTypeEmbeddingKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _computeTimeDiffMatrixKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>> _meanPoolRowsKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _embedWithBiasAndPositionalKernel;
        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int, float> _computeMemoryAttentionScoresKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>> _projectOutputBatchKernel;
        private Action<Index2D, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _embedTokenIdsKernel;
        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _meanPoolAllRowsKernel;
        private Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, float> _mat3DScaleInPlaceKernel;
        private Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView1D<float, Stride1D.Dense>> _matrixSquaredNorm3DKernel;
        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _softmaxVectorKernel;

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, int, int> _extractHeadQKVKernel;
        private Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float> _contentAwareScoresKernel;
        private Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _addDecayBiasKernel;
        private Action<Index1D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _contentAwareSoftmaxKernel;
        private Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _contentAwareWeightedSumKernel;
        private Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, int, int> _assembleHeadOutputKernel;

        private void InitTransformerSpecificKernels()
        {
            _applyContextTypeEmbeddingKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>>(ApplyContextTypeEmbeddingKernel);
            _computeTimeDiffMatrixKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(ComputeTimeDiffMatrixKernel);
            _meanPoolRowsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>>(MeanPoolRowsKernel);
            _embedWithBiasAndPositionalKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(EmbedWithBiasAndPositionalKernel);
            _computeMemoryAttentionScoresKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int, float>(ComputeMemoryAttentionScoresKernel);
            _projectOutputBatchKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>>(ProjectOutputBatchKernel);
            _embedTokenIdsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(EmbedTokenIdsKernel);
            _meanPoolAllRowsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(MeanPoolAllRowsKernel);
            _mat3DScaleInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, float>(Mat3DScaleInPlaceKernel);
            _matrixSquaredNorm3DKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView1D<float, Stride1D.Dense>>(MatrixSquaredNorm3DKernel);
            _softmaxVectorKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SoftmaxVectorKernel);

            _extractHeadQKVKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, int, int>(ExtractHeadQKVKernel);
            _contentAwareScoresKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float>(ContentAwareScoresKernel);
            _addDecayBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(AddDecayBiasKernel);
            _contentAwareSoftmaxKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(ContentAwareSoftmaxKernel);
            _contentAwareWeightedSumKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(ContentAwareWeightedSumKernel);
            _assembleHeadOutputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, int, int>(AssembleHeadOutputKernel);
        }

        private void DisposeTransformerSpecificBuffers()
        {
        }

        #region Transformer specific kernels

        private static void ApplyContextTypeEmbeddingKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> contextHidden, ArrayView2D<float, Stride2D.DenseX> typeEmbedding, ArrayView1D<int, Stride1D.Dense> typeIndices, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int d = idx.Y;
            int t = typeIndices[idx.X];
            result[idx] = contextHidden[idx] + typeEmbedding[t, d];
        }

        private static void ComputeTimeDiffMatrixKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> result, ArrayView1D<float, Stride1D.Dense> keyArrivalTimes)
        {
            int p = idx.X;
            int s = idx.Y;
            result[idx] = (float)p - keyArrivalTimes[s];
        }

        private static void MeanPoolRowsKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> hidden, ArrayView1D<int, Stride1D.Dense> storyOffsets, ArrayView1D<int, Stride1D.Dense> storyCounts, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int s = idx.X;
            int d = idx.Y;
            int start = storyOffsets[s];
            int count = storyCounts[s];

            if (count <= 0)
            {
                result[idx] = 0.0f;
                return;
            }

            float sum = 0.0f;
            for (int t = start; t < start + count; t++)
            {
                sum += hidden[t, d];
            }

            result[idx] = sum / count;
        }

        private static void EmbedWithBiasAndPositionalKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> projected, ArrayView1D<float, Stride1D.Dense> bias, ArrayView2D<float, Stride2D.DenseX> positionalEncoding, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int j = idx.Y;
            result[idx] = projected[idx] + bias[j] + positionalEncoding[idx];
        }

        private static void ComputeMemoryAttentionScoresKernel(Index1D s, ArrayView2D<float, Stride2D.DenseX> priceHidden, ArrayView2D<float, Stride2D.DenseX> contextHidden, ArrayView1D<float, Stride1D.Dense> scores, int lastPos, float scale)
        {
            int embDim = (int)priceHidden.Extent.Y;
            float dot = 0.0f;

            for (int d = 0; d < embDim; d++)
            {
                dot += priceHidden[lastPos, d] * contextHidden[s, d];
            }

            scores[s] = dot * scale;
        }

        private static void ProjectOutputBatchKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> hidden, ArrayView2D<float, Stride2D.DenseX> outputProjection, ArrayView1D<float, Stride1D.Dense> outputBias, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int i = idx.X;
            int j = idx.Y;
            int embDim = (int)hidden.Extent.Y;

            float sum = outputBias[j];
            for (int k = 0; k < embDim; k++)
            {
                sum += outputProjection[j, k] * hidden[i, k];
            }

            result[idx] = sum;
        }

        private static void EmbedTokenIdsKernel(Index2D idx, ArrayView1D<int, Stride1D.Dense> tokenIds, ArrayView2D<float, Stride2D.DenseX> embedding, ArrayView2D<float, Stride2D.DenseX> output)
        {
            int i = idx.X;
            int d = idx.Y;
            output[idx] = embedding[tokenIds[i], d];
        }

        private static void MeanPoolAllRowsKernel(Index1D d, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> output)
        {
            int rows = (int)matrix.Extent.X;
            if (rows == 0)
            {
                output[d] = 0.0f;
                return;
            }

            float sum = 0.0f;
            for (int r = 0; r < rows; r++)
            {
                sum += matrix[r, d];
            }

            output[d] = sum / rows;
        }

        private static void Mat3DScaleInPlaceKernel(Index3D idx, ArrayView3D<float, Stride3D.DenseXY> mat, float scale)
        {
            mat[idx] *= scale;
        }

        private static void MatrixSquaredNorm3DKernel(Index3D idx, ArrayView3D<float, Stride3D.DenseXY> matrix, ArrayView1D<float, Stride1D.Dense> sum)
        {
            float value = matrix[idx];
            Atomic.Add(ref sum[0], value * value);
        }

        private static void SoftmaxVectorKernel(Index1D _, ArrayView1D<float, Stride1D.Dense> scores, ArrayView1D<float, Stride1D.Dense> output)
        {
            int n = (int)scores.Extent.X;

            if (n == 0)
            {
                return;
            }

            float maxVal = float.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                float value = scores[i];
                if (value > maxVal)
                {
                    maxVal = value;
                }
            }

            float sum = 0.0f;
            for (int i = 0; i < n; i++)
            {
                float exp = XMath.Exp(scores[i] - maxVal);
                output[i] = exp;
                sum += exp;
            }

            if (sum <= 0.0f)
            {
                for (int i = 0; i < n; i++)
                {
                    output[i] = 0.0f;
                }

                return;
            }

            float inv = 1.0f / sum;
            for (int i = 0; i < n; i++)
            {
                output[i] *= inv;
            }
        }

        private static void ExtractHeadQKVKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> input, ArrayView3D<float, Stride3D.DenseXY> output, int headDim, int startIdx)
        {
            int i = idx.X;
            int j = idx.Y;
            output[0, i, j] = input[i, startIdx + j];
        }

        private static void ContentAwareScoresKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> Q_head, ArrayView3D<float, Stride3D.DenseXY> K_head, ArrayView3D<float, Stride3D.DenseXY> scores, float scale)
        {
            int i = idx.X;
            int j = idx.Y;
            int headDim = (int)Q_head.Extent.Z;

            float dot = 0.0f;
            for (int d = 0; d < headDim; d++)
            {
                dot += Q_head[0, i, d] * K_head[0, j, d];
            }

            scores[0, i, j] = dot * scale;
        }

        private static void AddDecayBiasKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> scores, ArrayView3D<float, Stride3D.DenseXY> decayBias, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int i = idx.X;
            int j = idx.Y;
            output[0, i, j] = scores[0, i, j] + decayBias[0, i, j];
        }

        private static void ContentAwareSoftmaxKernel(Index1D q, ArrayView3D<float, Stride3D.DenseXY> scores, ArrayView3D<float, Stride3D.DenseXY> weights)
        {
            int seqLenK = (int)scores.Extent.Z;

            float maxVal = float.NegativeInfinity;
            for (int k = 0; k < seqLenK; k++)
            {
                float value = scores[0, q, k];
                if (value > maxVal)
                {
                    maxVal = value;
                }
            }

            float sum = 0.0f;
            for (int k = 0; k < seqLenK; k++)
            {
                float exp = XMath.Exp(scores[0, q, k] - maxVal);
                weights[0, q, k] = exp;
                sum += exp;
            }

            float inv = 1.0f / XMath.Max(sum, 1e-20f);
            for (int k = 0; k < seqLenK; k++)
            {
                weights[0, q, k] *= inv;
            }
        }

        private static void ContentAwareWeightedSumKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView3D<float, Stride3D.DenseXY> V_head, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int i = idx.X;
            int d = idx.Y;
            int seqLenK = (int)weights.Extent.Z;

            float sum = 0.0f;
            for (int j = 0; j < seqLenK; j++)
            {
                sum += weights[0, i, j] * V_head[0, j, d];
            }

            output[0, i, d] = sum;
        }

        private static void AssembleHeadOutputKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> headOutput, ArrayView2D<float, Stride2D.DenseX> concatenated, int headDim, int startIdx)
        {
            int i = idx.X;
            int j = idx.Y;
            concatenated[i, startIdx + j] = headOutput[0, i, j];
        }

        #endregion

        #region Transformer specific - Multimodal/TACAMT/MMTAC

        public void ApplyContextTypeEmbedding(float[,] contextHidden, float[,] typeEmbedding, int[] typeIndices)
        {
            int n = contextHidden.GetLength(0);
            int embDim = contextHidden.GetLength(1);

            if (!ShouldUseGpu((long)n * embDim))
            {
                _mutliThreadCPU.ApplyContextTypeEmbedding(contextHidden, typeEmbedding, typeIndices);
                return;
            }

            var bufContext = _accelerator.Allocate2DDenseX<float>(new Index2D(n, embDim));
            var bufType = _accelerator.Allocate2DDenseX<float>(new Index2D(typeEmbedding.GetLength(0), embDim));
            var bufIndices = _accelerator.Allocate1D<int>(n);
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(n, embDim));

            try
            {
                bufContext.CopyFromCPU(contextHidden);
                bufType.CopyFromCPU(typeEmbedding);
                bufIndices.CopyFromCPU(typeIndices);

                _applyContextTypeEmbeddingKernel(new Index2D(n, embDim), bufContext.View, bufType.View, bufIndices.View, bufResult.View);
                bufResult.CopyToCPU(contextHidden);
            }
            finally
            {
                bufContext.Dispose();
                bufType.Dispose();
                bufIndices.Dispose();
                bufResult.Dispose();
            }
        }

        public float[,] ComputeTimeDiffMatrix(int priceSeqLen, float[] keyArrivalTimes)
        {
            int numKeys = keyArrivalTimes.Length;

            if (!ShouldUseGpu((long)priceSeqLen * numKeys))
            {
                return _mutliThreadCPU.ComputeTimeDiffMatrix(priceSeqLen, keyArrivalTimes);
            }

            var bufKeys = _accelerator.Allocate1D<float>(numKeys);
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(priceSeqLen, numKeys));

            try
            {
                bufKeys.CopyFromCPU(keyArrivalTimes);
                _computeTimeDiffMatrixKernel(new Index2D(priceSeqLen, numKeys), bufResult.View, bufKeys.View);

                var result = new float[priceSeqLen, numKeys];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufKeys.Dispose();
                bufResult.Dispose();
            }
        }

        public float[,] MeanPoolRows(float[,] hidden, int[] storyOffsets, int[] storyCounts, int numStories, int embeddingDim)
        {
            if (!ShouldUseGpu((long)numStories * embeddingDim))
            {
                var cpuResult = new float[numStories, embeddingDim];
                for (int s = 0; s < numStories; s++)
                {
                    int start = storyOffsets[s];
                    int count = storyCounts[s];
                    if (count <= 0)
                    {
                        continue;
                    }

                    for (int d = 0; d < embeddingDim; d++)
                    {
                        float sum = 0.0f;
                        for (int i = start; i < start + count; i++)
                        {
                            sum += hidden[i, d];
                        }

                        cpuResult[s, d] = sum / count;
                    }
                }

                return cpuResult;
            }

            var bufHidden = _accelerator.Allocate2DDenseX<float>(new Index2D(hidden.GetLength(0), embeddingDim));
            var bufOffsets = _accelerator.Allocate1D<int>(numStories);
            var bufCounts = _accelerator.Allocate1D<int>(numStories);
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(numStories, embeddingDim));

            try
            {
                bufHidden.CopyFromCPU(hidden);
                bufOffsets.CopyFromCPU(storyOffsets);
                bufCounts.CopyFromCPU(storyCounts);

                _meanPoolRowsKernel(new Index2D(numStories, embeddingDim), bufHidden.View, bufOffsets.View, bufCounts.View, bufResult.View);

                var result = new float[numStories, embeddingDim];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufHidden.Dispose();
                bufOffsets.Dispose();
                bufCounts.Dispose();
                bufResult.Dispose();
            }
        }

        public float[,] EmbedWithBiasAndPositional(float[,] projected, float[] bias, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            if (!ShouldUseGpu((long)seqLen * embeddingDim))
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

            var bufProj = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufBias = _accelerator.Allocate1D<float>(embeddingDim);
            var bufPos = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));

            try
            {
                bufProj.CopyFromCPU(projected);
                bufBias.CopyFromCPU(bias);
                bufPos.CopyFromCPU(positionalEncoding);

                _embedWithBiasAndPositionalKernel(new Index2D(seqLen, embeddingDim), bufProj.View, bufBias.View, bufPos.View, bufResult.View);

                var result = new float[seqLen, embeddingDim];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufProj.Dispose();
                bufBias.Dispose();
                bufPos.Dispose();
                bufResult.Dispose();
            }
        }

        public float[] ComputeMemoryAttentionScores(float[,] priceHidden, int lastPos, float[,] contextHidden, int totalCtx, float scale)
        {
            int embDim = priceHidden.GetLength(1);

            if (!ShouldUseGpu((long)totalCtx * embDim))
            {
                return _mutliThreadCPU.ComputeMemoryAttentionScores(priceHidden, lastPos, contextHidden, totalCtx, scale);
            }

            var bufPrice = _accelerator.Allocate2DDenseX<float>(new Index2D(priceHidden.GetLength(0), embDim));
            var bufContext = _accelerator.Allocate2DDenseX<float>(new Index2D(totalCtx, embDim));
            var bufScores = _accelerator.Allocate1D<float>(totalCtx);

            try
            {
                bufPrice.CopyFromCPU(priceHidden);
                bufContext.CopyFromCPU(contextHidden);

                _computeMemoryAttentionScoresKernel(new Index1D(totalCtx), bufPrice.View, bufContext.View, bufScores.View, lastPos, scale);

                var scores = new float[totalCtx];
                bufScores.CopyToCPU(scores);
                return scores;
            }
            finally
            {
                bufPrice.Dispose();
                bufContext.Dispose();
                bufScores.Dispose();
            }
        }

        public float[,] ProjectOutputBatch(float[,] hidden, float[,] outputProjection, float[] outputBias, int seqLen, int outputDim)
        {
            int embDim = hidden.GetLength(1);

            if (!ShouldUseGpu((long)seqLen * outputDim * embDim, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.ProjectOutputBatch(hidden, outputProjection, outputBias, seqLen, outputDim);
            }

            var bufHidden = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embDim));
            var bufProj = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, embDim));
            var bufBias = _accelerator.Allocate1D<float>(outputDim);
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, outputDim));

            try
            {
                bufHidden.CopyFromCPU(hidden);
                bufProj.CopyFromCPU(outputProjection);
                bufBias.CopyFromCPU(outputBias);

                _projectOutputBatchKernel(new Index2D(seqLen, outputDim), bufHidden.View, bufProj.View, bufBias.View, bufResult.View);

                var result = new float[seqLen, outputDim];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufHidden.Dispose();
                bufProj.Dispose();
                bufBias.Dispose();
                bufResult.Dispose();
            }
        }

        public (float[,,] decayBias, ContentAwareDecayCache cache) ContentAwareDecayForward(float[,] queryEmbeddings, float[,] keyEmbeddings, float[,] timeDiffs, float[] keyTimesFromRef, ContentAwareDecayNetwork network, bool isTraining = false, Random dropoutRng = null)
        {
            // The CPU path is kept here because the decay network uses host-side helpers such as ClampGate
            // and cache normalization details that must match AccelerationCPU exactly.
            return _mutliThreadCPU.ContentAwareDecayForward(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, network, isTraining, dropoutRng);
        }

        public float[,] ContentAwareCrossAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, float[,,] decayBias, out float[][,] attentionWeights, out float[][,] scoresPreSoftmax)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads");
            }

            if (!ShouldUseGpu((long)numHeads * seqLenQ * seqLenK * headDim, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.ContentAwareCrossAttentionForward(Q, K, V, numHeads, scale, decayBias, out attentionWeights, out scoresPreSoftmax);
            }

            attentionWeights = new float[numHeads][,];
            scoresPreSoftmax = new float[numHeads][,];
            var concatenated = new float[seqLenQ, embeddingDim];

            var bufQ = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenQ, embeddingDim));
            var bufK = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenK, embeddingDim));
            var bufV = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenK, embeddingDim));
            var bufConcatenated = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenQ, embeddingDim));

            var bufQHead = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, headDim));
            var bufKHead = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenK, headDim));
            var bufVHead = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenK, headDim));
            var bufScores = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, seqLenK));
            var bufScoresWithBias = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, seqLenK));
            var bufWeights = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, seqLenK));
            var bufHeadOutput = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, headDim));
            var bufDecayBias = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, seqLenK));

            try
            {
                bufQ.CopyFromCPU(Q);
                bufK.CopyFromCPU(K);
                bufV.CopyFromCPU(V);

                for (int head = 0; head < numHeads; head++)
                {
                    int startIdx = head * headDim;
                    bufWeights.MemSetToZero();

                    _extractHeadQKVKernel(new Index2D(seqLenQ, headDim), bufQ.View, bufQHead.View, headDim, startIdx);
                    _extractHeadQKVKernel(new Index2D(seqLenK, headDim), bufK.View, bufKHead.View, headDim, startIdx);
                    _extractHeadQKVKernel(new Index2D(seqLenK, headDim), bufV.View, bufVHead.View, headDim, startIdx);

                    _contentAwareScoresKernel(new Index2D(seqLenQ, seqLenK), bufQHead.View, bufKHead.View, bufScores.View, scale);

                    if (decayBias != null)
                    {
                        var decayBiasHead = new float[1, seqLenQ, seqLenK];
                        for (int i = 0; i < seqLenQ; i++)
                        {
                            for (int j = 0; j < seqLenK; j++)
                            {
                                decayBiasHead[0, i, j] = decayBias[i, j, head];
                            }
                        }

                        bufDecayBias.CopyFromCPU(decayBiasHead);
                        _addDecayBiasKernel(new Index2D(seqLenQ, seqLenK), bufScores.View, bufDecayBias.View, bufScoresWithBias.View);
                    }
                    else
                    {
                        bufScores.CopyTo(bufScoresWithBias);
                    }

                    _contentAwareSoftmaxKernel(new Index1D(seqLenQ), bufScoresWithBias.View, bufWeights.View);
                    _contentAwareWeightedSumKernel(new Index2D(seqLenQ, headDim), bufWeights.View, bufVHead.View, bufHeadOutput.View);
                    _assembleHeadOutputKernel(new Index2D(seqLenQ, headDim), bufHeadOutput.View, bufConcatenated.View, headDim, startIdx);

                    var weightsHead = new float[1, seqLenQ, seqLenK];
                    var scoresHead = new float[1, seqLenQ, seqLenK];
                    bufWeights.CopyToCPU(weightsHead);
                    bufScoresWithBias.CopyToCPU(scoresHead);

                    attentionWeights[head] = new float[seqLenQ, seqLenK];
                    scoresPreSoftmax[head] = new float[seqLenQ, seqLenK];

                    for (int i = 0; i < seqLenQ; i++)
                    {
                        for (int j = 0; j < seqLenK; j++)
                        {
                            attentionWeights[head][i, j] = weightsHead[0, i, j];
                            scoresPreSoftmax[head][i, j] = scoresHead[0, i, j];
                        }
                    }
                }

                bufConcatenated.CopyToCPU(concatenated);
                return concatenated;
            }
            finally
            {
                bufQ.Dispose();
                bufK.Dispose();
                bufV.Dispose();
                bufConcatenated.Dispose();
                bufQHead.Dispose();
                bufKHead.Dispose();
                bufVHead.Dispose();
                bufScores.Dispose();
                bufScoresWithBias.Dispose();
                bufWeights.Dispose();
                bufHeadOutput.Dispose();
                bufDecayBias.Dispose();
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
            TacamtBlock block,
            BlockCache bc,
            int PriceEmbeddingDim,
            int PriceNumHeads,
            bool enableDecayBias = true,
            bool isTraining = false,
            Random dropoutRng = null)
        {
            int psl = Q.GetLength(0);
            int tsl = K.GetLength(0);
            long workUnits = (long)psl * tsl * PriceEmbeddingDim;

            if (!ShouldUseGpu(workUnits, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.ContentAwareCrossAttentionWithCache(Q, K, V, timeDiffs, keyTimesFromRef, queryEmbeddings, keyEmbeddings, block, bc, PriceEmbeddingDim, PriceNumHeads, enableDecayBias, isTraining, dropoutRng);
            }

            int headDim = PriceEmbeddingDim / PriceNumHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);
            float[,,] decayBias = null;

            if (enableDecayBias && timeDiffs != null)
            {
                var (bias, decayCache) = ContentAwareDecayForward(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, block.DecayNetwork, isTraining, dropoutRng);
                decayBias = bias;
                bc.DecayCache = decayCache;
            }
            else
            {
                bc.DecayCache = null;
            }

            var output = ContentAwareCrossAttentionForward(Q, K, V, PriceNumHeads, scale, decayBias, out var attentionWeights, out var scoresPreSoftmax);
            bc.CrossAttentionWeights = attentionWeights;
            bc.CrossScoresPreSoftmax = scoresPreSoftmax;
            return output;
        }

        public void Matrix3DScaleInPlace(float[,,] matrix, float scale)
        {
            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);

            if (!ShouldUseGpu((long)d0 * d1 * d2))
            {
                _mutliThreadCPU.Matrix3DScaleInPlace(matrix, scale);
                return;
            }

            var buf = _accelerator.Allocate3DDenseXY<float>(new Index3D(d0, d1, d2));
            try
            {
                buf.CopyFromCPU(matrix);
                _mat3DScaleInPlaceKernel(new Index3D(d0, d1, d2), buf.View, scale);
                buf.CopyToCPU(matrix);
            }
            finally
            {
                buf.Dispose();
            }
        }

        public float MatrixSquaredNorm3D(float[,,] matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);

            if (!ShouldUseGpu((long)d0 * d1 * d2))
            {
                return _mutliThreadCPU.MatrixSquaredNorm3D(matrix);
            }

            var bufMatrix = _accelerator.Allocate3DDenseXY<float>(new Index3D(d0, d1, d2));
            var bufSum = _accelerator.Allocate1D<float>(1);

            try
            {
                bufMatrix.CopyFromCPU(matrix);
                bufSum.CopyFromCPU(new float[1]);
                _matrixSquaredNorm3DKernel(new Index3D(d0, d1, d2), bufMatrix.View, bufSum.View);

                var result = new float[1];
                bufSum.CopyToCPU(result);
                return result[0];
            }
            finally
            {
                bufMatrix.Dispose();
                bufSum.Dispose();
            }
        }

        public float[] ProjectGlobalFeatures(float[] globalFeatures, float[,] projection, float[] bias)
        {
            if (projection == null) throw new ArgumentNullException(nameof(projection));
            if (globalFeatures == null) throw new ArgumentNullException(nameof(globalFeatures));
            if (bias == null) throw new ArgumentNullException(nameof(bias));

            int outputDim = projection.GetLength(0);
            int inputDim = projection.GetLength(1);

            if (!ShouldUseGpu((long)outputDim * inputDim, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.ProjectGlobalFeatures(globalFeatures, projection, bias);
            }

            var result = CalculateDotProduct(projection, globalFeatures);
            for (int i = 0; i < result.Length; i++)
            {
                result[i] += bias[i];
            }

            return result;
        }

        public float[,] EmbedTokenIds(int[] tokenIds, float[,] embedding, int embeddingDim)
        {
            if (tokenIds == null || tokenIds.Length == 0)
            {
                return new float[0, embeddingDim];
            }

            if (!ShouldUseGpu((long)tokenIds.Length * embeddingDim))
            {
                return _mutliThreadCPU.EmbedTokenIds(tokenIds, embedding, embeddingDim);
            }

            var bufTokenIds = _accelerator.Allocate1D<int>(tokenIds.Length);
            var bufEmbedding = _accelerator.Allocate2DDenseX<float>(new Index2D(embedding.GetLength(0), embeddingDim));
            var bufOutput = _accelerator.Allocate2DDenseX<float>(new Index2D(tokenIds.Length, embeddingDim));

            try
            {
                bufTokenIds.CopyFromCPU(tokenIds);
                bufEmbedding.CopyFromCPU(embedding);
                _embedTokenIdsKernel(new Index2D(tokenIds.Length, embeddingDim), bufTokenIds.View, bufEmbedding.View, bufOutput.View);

                var result = new float[tokenIds.Length, embeddingDim];
                bufOutput.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufTokenIds.Dispose();
                bufEmbedding.Dispose();
                bufOutput.Dispose();
            }
        }

        public float[] MeanPoolRows(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                return _mutliThreadCPU.MeanPoolRows(matrix);
            }

            var bufMatrix = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufOutput = _accelerator.Allocate1D<float>(cols);

            try
            {
                bufMatrix.CopyFromCPU(matrix);
                _meanPoolAllRowsKernel(new Index1D(cols), bufMatrix.View, bufOutput.View);

                var result = new float[cols];
                bufOutput.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufMatrix.Dispose();
                bufOutput.Dispose();
            }
        }

        public (float[,] contextHidden, float[] contextTimes, int numGlobal, int numNews) BuildMmtacContext(float[,] newsHidden, float[] newsTimes, float[] globalToken, float[,] contextTypeEmbedding)
        {
            if (contextTypeEmbedding == null)
            {
                throw new ArgumentNullException(nameof(contextTypeEmbedding));
            }

            int embeddingDim = contextTypeEmbedding.GetLength(1);
            int numGlobal = globalToken != null ? 1 : 0;
            int numNews = newsHidden != null ? newsHidden.GetLength(0) : 0;
            int total = numGlobal + numNews;

            if (total == 0)
            {
                return (null, null, 0, 0);
            }

            if (globalToken != null && globalToken.Length != embeddingDim)
            {
                throw new ArgumentException("globalToken length must match embedding dimension.", nameof(globalToken));
            }

            if (newsHidden != null && newsHidden.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("newsHidden embedding dimension mismatch.", nameof(newsHidden));
            }

            if (newsTimes != null && newsTimes.Length != numNews)
            {
                throw new ArgumentException("newsTimes length must match newsHidden row count.", nameof(newsTimes));
            }

            var contextHidden = new float[total, embeddingDim];
            var contextTimes = new float[total];
            int row = 0;

            if (globalToken != null)
            {
                for (int d = 0; d < embeddingDim; d++)
                {
                    contextHidden[row, d] = globalToken[d] + contextTypeEmbedding[2, d];
                }

                contextTimes[row] = 0.0f;
                row++;
            }

            for (int i = 0; i < numNews; i++)
            {
                for (int d = 0; d < embeddingDim; d++)
                {
                    contextHidden[row, d] = newsHidden[i, d] + contextTypeEmbedding[0, d];
                }

                contextTimes[row] = newsTimes != null ? newsTimes[i] : 0.0f;
                row++;
            }

            return (contextHidden, contextTimes, numGlobal, numNews);
        }

        public (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence, float[,] regressionLogits, float[] rangeLogits, float[] qualityLogits) ProjectMmtacOutputHeads(
            float[,] hidden,
            float[,] regressionProjection,
            float[] regressionBias,
            float[,] rangeProjection,
            float[] rangeBias,
            float[,] qualityProjection,
            float[] qualityBias,
            float[,] directionProjection,
            float[] directionBias,
            float[,] midDirectionProjection,
            float[] midDirectionBias,
            float[,] confidenceProjection,
            float[] confidenceBias,
            bool useConfidenceHead)
        {
            if (hidden == null)
            {
                throw new ArgumentNullException(nameof(hidden));
            }

            int seqLen = hidden.GetLength(0);
            const int regressionDim = 3;

            var rawRegression = ProjectOutputBatch(hidden, regressionProjection, regressionBias, seqLen, regressionDim);
            var rawRange = ProjectOutputBatch(hidden, rangeProjection, rangeBias, seqLen, 1);
            var rawQuality = ProjectOutputBatch(hidden, qualityProjection, qualityBias, seqLen, 1);
            var rawDirection = ProjectOutputBatch(hidden, directionProjection, directionBias, seqLen, 1);
            var rawMidDirection = ProjectOutputBatch(hidden, midDirectionProjection, midDirectionBias, seqLen, 1);
            float[,] rawConfidence = null;

            if (useConfidenceHead)
            {
                rawConfidence = ProjectOutputBatch(hidden, confidenceProjection, confidenceBias, seqLen, 1);
            }

            var regression = new float[seqLen, regressionDim];
            var range = new float[seqLen, 1];
            var rangeLogits = new float[seqLen];
            var quality = new float[seqLen, 1];
            var qualityLogits = new float[seqLen];
            var direction = new float[seqLen, 1];
            var midDirection = new float[seqLen, 1];
            float[,] confidence = useConfidenceHead ? new float[seqLen, 1] : null;

            for (int t = 0; t < seqLen; t++)
            {
                float upBase = SoftplusHost(rawRegression[t, 0]);
                float downBase = SoftplusHost(rawRegression[t, 1]);
                float rangeLogit = rawRange[t, 0];
                float rangeValue = SoftplusHost(rangeLogit);
                float den = upBase + downBase;
                float upShare = den > 1e-6f ? upBase / den : 0.5f;
                float downShare = 1.0f - upShare;
                float close = rawRegression[t, 2];

                regression[t, 0] = close + rangeValue * upShare;
                regression[t, 1] = close - rangeValue * downShare;
                regression[t, 2] = close;
                range[t, 0] = rangeValue;
                rangeLogits[t] = rangeLogit;

                float qualityLogit = rawQuality[t, 0];
                qualityLogits[t] = qualityLogit;
                quality[t, 0] = StableSigmoidHost(qualityLogit);
                direction[t, 0] = StableSigmoidHost(rawDirection[t, 0]);
                midDirection[t, 0] = StableSigmoidHost(rawMidDirection[t, 0]);

                if (useConfidenceHead)
                {
                    confidence[t, 0] = StableSigmoidHost(rawConfidence[t, 0]);
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

            int n = scores.Length;
            if (n == 0)
            {
                return Array.Empty<float>();
            }

            if (!ShouldUseGpu(n))
            {
                return _mutliThreadCPU.SoftmaxVector(scores);
            }

            var bufScores = _accelerator.Allocate1D<float>(n);
            var bufOutput = _accelerator.Allocate1D<float>(n);

            try
            {
                bufScores.CopyFromCPU(scores);
                _softmaxVectorKernel(new Index1D(1), bufScores.View, bufOutput.View);

                var output = new float[n];
                bufOutput.CopyToCPU(output);
                return output;
            }
            finally
            {
                bufScores.Dispose();
                bufOutput.Dispose();
            }
        }

        private static float StableSigmoidHost(float x)
        {
            if (x >= 0.0f)
            {
                float ex = MathF.Exp(-x);
                return 1.0f / (1.0f + ex);
            }

            float exp = MathF.Exp(x);
            return exp / (1.0f + exp);
        }

        private static float SoftplusHost(float x)
        {
            if (x > 20.0f)
            {
                return x;
            }

            if (x < -20.0f)
            {
                return MathF.Exp(x);
            }

            return MathF.Log(1.0f + MathF.Exp(x));
        }

        #endregion
    }
}
