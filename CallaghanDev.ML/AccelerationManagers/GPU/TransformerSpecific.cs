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
        private Action<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int> _buildMmtacContextWithPriceHiddenKernel;
        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int> _buildMmtacContextWithPriceTimesKernel;

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, int, int, int, int, int> _accumulateMmtacContextTypeGradKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, int, int, int> _accumulateMmtacLiveNewsGradKernel;
        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int> _accumulateMmtacGlobalGradKernel;

        private Action<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _accumulateGlobalProjectionGradKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int> _expandMeanPoolGradientKernel;

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, float> _mmtacScalarMseGradKernel;
        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, float> _mmtacScalarBceGradKernel;
        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, float, int, int> _mmtacConfidenceGradKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int> _mmtacOutputDHiddenKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _mmtacOutputProjectionGradKernel;
        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _mmtacOutputBiasGradKernel;

        private Action<
    Index1D,
    ArrayView2D<float, Stride2D.DenseX>,
    ArrayView2D<float, Stride2D.DenseX>,
    ArrayView2D<float, Stride2D.DenseX>,
    ArrayView1D<float, Stride1D.Dense>,
    ArrayView2D<float, Stride2D.DenseX>,
    ArrayView1D<float, Stride1D.Dense>,
    float,
    float,
    int> _mmtacRegressionBaseGradKernel;

        private Action<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            float> _mmtacRangeShareGradKernel;
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
            _mat3DAddInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(Mat3DAddInPlaceKernel);
            _mat3DUpdateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float>(Mat3DUpdateKernel);
            _matrixSquaredNorm3DKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView1D<float, Stride1D.Dense>>(MatrixSquaredNorm3DKernel);
            _softmaxVectorKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SoftmaxVectorKernel);

            _extractHeadQKVKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, int, int>(ExtractHeadQKVKernel);
            _contentAwareScoresKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float>(ContentAwareScoresKernel);
            _addDecayBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(AddDecayBiasKernel);
            _contentAwareSoftmaxKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(ContentAwareSoftmaxKernel);
            _contentAwareWeightedSumKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(ContentAwareWeightedSumKernel);
            _assembleHeadOutputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, int, int>(AssembleHeadOutputKernel);

            _buildMmtacContextWithPriceHiddenKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int>(BuildMmtacContextWithPriceHiddenKernel);
            _buildMmtacContextWithPriceTimesKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(BuildMmtacContextWithPriceTimesKernel);

            _accumulateMmtacContextTypeGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, int, int, int, int, int>(AccumulateMmtacContextTypeGradKernel);
            _accumulateMmtacLiveNewsGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, int, int, int>(AccumulateMmtacLiveNewsGradKernel);
            _accumulateMmtacGlobalGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(AccumulateMmtacGlobalGradKernel);

            _accumulateGlobalProjectionGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(AccumulateGlobalProjectionGradKernel);
            _expandMeanPoolGradientKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int>(ExpandMeanPoolGradientKernel);

            _mmtacRegressionBaseGradKernel =
       _accelerator.LoadAutoGroupedStreamKernel<
           Index1D,
           ArrayView2D<float, Stride2D.DenseX>,
           ArrayView2D<float, Stride2D.DenseX>,
           ArrayView2D<float, Stride2D.DenseX>,
           ArrayView1D<float, Stride1D.Dense>,
           ArrayView2D<float, Stride2D.DenseX>,
           ArrayView1D<float, Stride1D.Dense>,
           float,
           float,
           int>(
               MmtacRegressionBaseGradKernel);

            _mmtacRangeShareGradKernel =
                _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView1D<float, Stride1D.Dense>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView1D<float, Stride1D.Dense>,
                    float>(
                        MmtacRangeShareGradKernel); _mmtacScalarMseGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, float>(MmtacScalarMseGradKernel);
            _mmtacScalarBceGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, float>(MmtacScalarBceGradKernel);
            _mmtacConfidenceGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, float, int, int>(MmtacConfidenceGradKernel);
            _mmtacOutputDHiddenKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(MmtacOutputDHiddenKernel);
            _mmtacOutputProjectionGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(MmtacOutputProjectionGradKernel);
            _mmtacOutputBiasGradKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(MmtacOutputBiasGradKernel);

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



        private Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _mat3DAddInPlaceKernel;
        private static void Mat3DAddInPlaceKernel(Index3D idx, ArrayView3D<float, Stride3D.DenseXY> target, ArrayView3D<float, Stride3D.DenseXY> addend)
        {
            target[idx] += addend[idx];
        }
        public void Matrix3DAddInPlace(float[,,] target, float[,,] addend)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (addend == null) throw new ArgumentNullException(nameof(addend));

            int d0 = target.GetLength(0);
            int d1 = target.GetLength(1);
            int d2 = target.GetLength(2);

            if (!ShouldUseGpu((long)d0 * d1 * d2))
            {
                _mutliThreadCPU.Matrix3DAddInPlace(target, addend);
                return;
            }

            var bufTarget = _accelerator.Allocate3DDenseXY<float>(new Index3D(d0, d1, d2));
            var bufAddend = _accelerator.Allocate3DDenseXY<float>(new Index3D(d0, d1, d2));

            try
            {
                bufTarget.CopyFromCPU(target);
                bufAddend.CopyFromCPU(addend);
                _mat3DAddInPlaceKernel(new Index3D(d0, d1, d2), bufTarget.View, bufAddend.View);
                bufTarget.CopyToCPU(target);
            }
            finally
            {
                bufTarget.Dispose();
                bufAddend.Dispose();
            }
        }


        private Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float> _mat3DUpdateKernel;
        private static void Mat3DUpdateKernel(Index3D idx, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView3D<float, Stride3D.DenseXY> gradients, float learningRate)
        {
            weights[idx] -= learningRate * gradients[idx];
        }
        public void Matrix3DUpdate(float[,,] weights, float[,,] gradients, float learningRate)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (gradients == null) throw new ArgumentNullException(nameof(gradients));

            int d0 = weights.GetLength(0);
            int d1 = weights.GetLength(1);
            int d2 = weights.GetLength(2);

            if (!ShouldUseGpu((long)d0 * d1 * d2))
            {
                _mutliThreadCPU.Matrix3DUpdate(weights, gradients, learningRate);
                return;
            }

            var bufWeights = _accelerator.Allocate3DDenseXY<float>(new Index3D(d0, d1, d2));
            var bufGradients = _accelerator.Allocate3DDenseXY<float>(new Index3D(d0, d1, d2));

            try
            {
                bufWeights.CopyFromCPU(weights);
                bufGradients.CopyFromCPU(gradients);
                _mat3DUpdateKernel(new Index3D(d0, d1, d2), bufWeights.View, bufGradients.View, learningRate);
                bufWeights.CopyToCPU(weights);
            }
            finally
            {
                bufWeights.Dispose();
                bufGradients.Dispose();
            }
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

            if (priceContextHidden != null && priceContextHidden.GetLength(1) != ed)
            {
                throw new ArgumentException("priceContextHidden embedding dimension mismatch.", nameof(priceContextHidden));
            }

            if (newsTimes != null && newsTimes.Length != numNews)
            {
                throw new ArgumentException("newsTimes length must match newsHidden row count.", nameof(newsTimes));
            }

            if (priceContextTimes != null && priceContextTimes.Length != numPrice)
            {
                throw new ArgumentException("priceContextTimes length must match priceContextHidden row count.", nameof(priceContextTimes));
            }

            if (!ShouldUseGpu((long)total * ed))
            {
                return _mutliThreadCPU.BuildMmtacContextWithPrice(newsHidden, newsTimes, globalToken, priceContextHidden, priceContextTimes, contextTypeEmbedding);
            }

            var bufGlobal = _accelerator.Allocate1D<float>(Math.Max(1, ed));
            var bufNews = _accelerator.Allocate2DDenseX<float>(new Index2D(Math.Max(1, numNews), ed));
            var bufPrice = _accelerator.Allocate2DDenseX<float>(new Index2D(Math.Max(1, numPrice), ed));
            var bufType = _accelerator.Allocate2DDenseX<float>(new Index2D(contextTypeEmbedding.GetLength(0), ed));
            var bufHidden = _accelerator.Allocate2DDenseX<float>(new Index2D(total, ed));
            var bufNewsTimes = _accelerator.Allocate1D<float>(Math.Max(1, numNews));
            var bufPriceTimes = _accelerator.Allocate1D<float>(Math.Max(1, numPrice));
            var bufTimes = _accelerator.Allocate1D<float>(total);

            try
            {
                bufGlobal.CopyFromCPU(globalToken ?? new float[Math.Max(1, ed)]);
                bufNews.CopyFromCPU(newsHidden ?? new float[Math.Max(1, numNews), ed]);
                bufPrice.CopyFromCPU(priceContextHidden ?? new float[Math.Max(1, numPrice), ed]);
                bufType.CopyFromCPU(contextTypeEmbedding);
                bufNewsTimes.CopyFromCPU(newsTimes ?? new float[Math.Max(1, numNews)]);
                bufPriceTimes.CopyFromCPU(priceContextTimes ?? new float[Math.Max(1, numPrice)]);

                _buildMmtacContextWithPriceHiddenKernel(new Index2D(total, ed), bufGlobal.View, bufNews.View, bufPrice.View, bufType.View, bufHidden.View, numGlobal, numNews);
                _buildMmtacContextWithPriceTimesKernel(new Index1D(total), bufNewsTimes.View, bufPriceTimes.View, bufTimes.View, numGlobal, numNews, newsTimes != null ? 1 : 0, priceContextTimes != null ? 1 : 0);

                var contextHidden = new float[total, ed];
                var contextTimes = new float[total];
                bufHidden.CopyToCPU(contextHidden);
                bufTimes.CopyToCPU(contextTimes);
                return (contextHidden, contextTimes, numGlobal, numNews, numPrice);
            }
            finally
            {
                bufGlobal.Dispose();
                bufNews.Dispose();
                bufPrice.Dispose();
                bufType.Dispose();
                bufHidden.Dispose();
                bufNewsTimes.Dispose();
                bufPriceTimes.Dispose();
                bufTimes.Dispose();
            }
        }
        public (float loss, float[,] dHidden) BackpropMmtacOutputHeads(
                   float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence,
                   float[,] targetRegression, float[,] targetRange, float[,] targetQuality, float[,] targetDirection, float[,] targetMidDirection,
                   float[] previousClose, float[] confidenceTargets,
                   float[,] hidden, float[,] regressionLogits, float[] rangeLogits,
                   float[,] regressionProjection, float[,] rangeProjection, float[,] qualityProjection, float[,] directionProjection, float[,] midDirectionProjection, float[,] confidenceProjection,
                   float[,] regressionProjectionGrad, float[] regressionBiasGrad,
                   float[,] rangeProjectionGrad, float[] rangeBiasGrad,
                   float[,] qualityProjectionGrad, float[] qualityBiasGrad,
                   float[,] directionProjectionGrad, float[] directionBiasGrad,
                   float[,] midDirectionProjectionGrad, float[] midDirectionBiasGrad,
                   float[,] confidenceProjectionGrad, float[] confidenceBiasGrad,
                   float rangeLossWeight, float qualityLossWeight, float directionLossWeight, float midDirectionLossWeight,
                   float closeDirectionConsistencyWeight, float closeDirectionConsistencyMargin,
                   float confidenceLossWeight, bool useConfidenceHead)
        {
            if (regression == null) throw new ArgumentNullException(nameof(regression));
            if (range == null) throw new ArgumentNullException(nameof(range));
            if (quality == null) throw new ArgumentNullException(nameof(quality));
            if (direction == null) throw new ArgumentNullException(nameof(direction));
            if (midDirection == null) throw new ArgumentNullException(nameof(midDirection));
            if (targetRegression == null) throw new ArgumentNullException(nameof(targetRegression));
            if (targetRange == null) throw new ArgumentNullException(nameof(targetRange));
            if (targetQuality == null) throw new ArgumentNullException(nameof(targetQuality));
            if (targetDirection == null) throw new ArgumentNullException(nameof(targetDirection));
            if (targetMidDirection == null) throw new ArgumentNullException(nameof(targetMidDirection));
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

            float effectiveConfidenceWeight = useConfidenceHead ? MathF.Max(0.0f, confidenceLossWeight) : 0.0f;
            bool useConfidence = useConfidenceHead
                && confidence != null
                && confidenceProjection != null
                && confidenceProjectionGrad != null
                && confidenceBiasGrad != null
                && effectiveConfidenceWeight > 0.0f;

            float[] rangeLogitsHost = rangeLogits;
            if (rangeLogitsHost.Length != sl)
            {
                var tmp = new float[sl];
                Array.Copy(rangeLogitsHost, tmp, Math.Min(sl, rangeLogitsHost.Length));
                rangeLogitsHost = tmp;
            }

            float[] previousCloseHost = previousClose ?? new float[sl];
            if (previousCloseHost.Length != sl)
            {
                var tmp = new float[sl];
                Array.Copy(previousCloseHost, tmp, Math.Min(sl, previousCloseHost.Length));
                previousCloseHost = tmp;
            }

            float[] confidenceTargetsHost = confidenceTargets ?? new float[sl];
            if (confidenceTargetsHost.Length != sl)
            {
                var tmp = new float[sl];
                Array.Copy(confidenceTargetsHost, tmp, Math.Min(sl, confidenceTargetsHost.Length));
                confidenceTargetsHost = tmp;
            }

            long workUnits = (long)sl * (ed * (rDim + 5) + rDim + 8);
            if (!ShouldUseGpu(workUnits, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.BackpropMmtacOutputHeads(
                    regression, range, quality, direction, midDirection, confidence,
                    targetRegression, targetRange, targetQuality, targetDirection, targetMidDirection,
                    previousClose, confidenceTargets,
                    hidden, regressionLogits, rangeLogits,
                    regressionProjection, rangeProjection, qualityProjection, directionProjection, midDirectionProjection, confidenceProjection,
                    regressionProjectionGrad, regressionBiasGrad,
                    rangeProjectionGrad, rangeBiasGrad,
                    qualityProjectionGrad, qualityBiasGrad,
                    directionProjectionGrad, directionBiasGrad,
                    midDirectionProjectionGrad, midDirectionBiasGrad,
                    confidenceProjectionGrad, confidenceBiasGrad,
                    rangeLossWeight, qualityLossWeight, directionLossWeight, midDirectionLossWeight,
                    closeDirectionConsistencyWeight, closeDirectionConsistencyMargin,
                    confidenceLossWeight, useConfidenceHead);
            }

            var bufRegression = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, rDim));
            var bufRange = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufQuality = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufDirection = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufMidDirection = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufConfidence = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));

            var bufTargetRegression = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, rDim));
            var bufTargetRange = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufTargetQuality = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufTargetDirection = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufTargetMidDirection = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));

            var bufPreviousClose = _accelerator.Allocate1D<float>(sl);
            var bufConfidenceTargets = _accelerator.Allocate1D<float>(sl);
            var bufHidden = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, ed));
            var bufRegressionLogits = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, rDim));
            var bufRangeLogits = _accelerator.Allocate1D<float>(sl);

            var bufRegressionProjection = _accelerator.Allocate2DDenseX<float>(new Index2D(rDim, ed));
            var bufRangeProjection = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));
            var bufQualityProjection = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));
            var bufDirectionProjection = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));
            var bufMidDirectionProjection = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));
            var bufConfidenceProjection = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));

            var bufRegressionProjectionGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(rDim, ed));
            var bufRangeProjectionGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));
            var bufQualityProjectionGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));
            var bufDirectionProjectionGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));
            var bufMidDirectionProjectionGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));
            var bufConfidenceProjectionGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(1, ed));

            var bufRegressionBiasGrad = _accelerator.Allocate1D<float>(rDim);
            var bufRangeBiasGrad = _accelerator.Allocate1D<float>(1);
            var bufQualityBiasGrad = _accelerator.Allocate1D<float>(1);
            var bufDirectionBiasGrad = _accelerator.Allocate1D<float>(1);
            var bufMidDirectionBiasGrad = _accelerator.Allocate1D<float>(1);
            var bufConfidenceBiasGrad = _accelerator.Allocate1D<float>(1);

            var bufDRegression = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, rDim));
            var bufDRange = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufDQuality = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufDDirection = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufDMidDirection = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufDConfidence = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, 1));
            var bufDHidden = _accelerator.Allocate2DDenseX<float>(new Index2D(sl, ed));
            var bufLoss = _accelerator.Allocate1D<float>(1);

            try
            {
                bufRegression.CopyFromCPU(regression);
                bufRange.CopyFromCPU(range);
                bufQuality.CopyFromCPU(quality);
                bufDirection.CopyFromCPU(direction);
                bufMidDirection.CopyFromCPU(midDirection);
                bufConfidence.CopyFromCPU(useConfidence ? confidence : new float[sl, 1]);

                bufTargetRegression.CopyFromCPU(targetRegression);
                bufTargetRange.CopyFromCPU(targetRange);
                bufTargetQuality.CopyFromCPU(targetQuality);
                bufTargetDirection.CopyFromCPU(targetDirection);
                bufTargetMidDirection.CopyFromCPU(targetMidDirection);

                bufPreviousClose.CopyFromCPU(previousCloseHost);
                bufConfidenceTargets.CopyFromCPU(confidenceTargetsHost);
                bufHidden.CopyFromCPU(hidden);
                bufRegressionLogits.CopyFromCPU(regressionLogits);
                bufRangeLogits.CopyFromCPU(rangeLogitsHost);

                bufRegressionProjection.CopyFromCPU(regressionProjection);
                bufRangeProjection.CopyFromCPU(rangeProjection);
                bufQualityProjection.CopyFromCPU(qualityProjection);
                bufDirectionProjection.CopyFromCPU(directionProjection);
                bufMidDirectionProjection.CopyFromCPU(midDirectionProjection);
                bufConfidenceProjection.CopyFromCPU(useConfidence ? confidenceProjection : new float[1, ed]);

                bufRegressionProjectionGrad.CopyFromCPU(regressionProjectionGrad);
                bufRangeProjectionGrad.CopyFromCPU(rangeProjectionGrad);
                bufQualityProjectionGrad.CopyFromCPU(qualityProjectionGrad);
                bufDirectionProjectionGrad.CopyFromCPU(directionProjectionGrad);
                bufMidDirectionProjectionGrad.CopyFromCPU(midDirectionProjectionGrad);
                bufConfidenceProjectionGrad.CopyFromCPU(useConfidence ? confidenceProjectionGrad : new float[1, ed]);

                bufRegressionBiasGrad.CopyFromCPU(regressionBiasGrad);
                bufRangeBiasGrad.CopyFromCPU(rangeBiasGrad);
                bufQualityBiasGrad.CopyFromCPU(qualityBiasGrad);
                bufDirectionBiasGrad.CopyFromCPU(directionBiasGrad);
                bufMidDirectionBiasGrad.CopyFromCPU(midDirectionBiasGrad);
                bufConfidenceBiasGrad.CopyFromCPU(useConfidence ? confidenceBiasGrad : new float[1]);

                bufDRegression.MemSetToZero();
                bufDRange.MemSetToZero();
                bufDQuality.MemSetToZero();
                bufDDirection.MemSetToZero();
                bufDMidDirection.MemSetToZero();
                bufDConfidence.MemSetToZero();
                bufDHidden.MemSetToZero();
                bufLoss.CopyFromCPU(new float[1]);

                _mmtacRegressionBaseGradKernel(
                     new Index1D(sl),
                     bufRegression.View,
                     bufTargetRegression.View,
                     bufTargetDirection.View,
                     bufPreviousClose.View,
                     bufDRegression.View,
                     bufLoss.View,
                     closeDirectionConsistencyWeight,
                     closeDirectionConsistencyMargin,
                     previousClose != null ? 1 : 0);

                _mmtacRangeShareGradKernel(
                    new Index1D(sl),
                    bufRange.View,
                    bufTargetRange.View,
                    bufRegressionLogits.View,
                    bufRangeLogits.View,
                    bufDRegression.View,
                    bufDRange.View,
                    bufLoss.View,
                    rangeLossWeight);
                
                _mmtacScalarMseGradKernel(new Index1D(sl), bufQuality.View, bufTargetQuality.View, bufDQuality.View, bufLoss.View, qualityLossWeight);
                _mmtacScalarBceGradKernel(new Index1D(sl), bufDirection.View, bufTargetDirection.View, bufDDirection.View, bufLoss.View, directionLossWeight);
                _mmtacScalarBceGradKernel(new Index1D(sl), bufMidDirection.View, bufTargetMidDirection.View, bufDMidDirection.View, bufLoss.View, midDirectionLossWeight);

                if (useConfidence)
                {
                    _mmtacConfidenceGradKernel(new Index1D(sl), bufConfidence.View, bufConfidenceTargets.View, bufRegression.View, bufTargetRegression.View, bufDConfidence.View, bufLoss.View, effectiveConfidenceWeight, confidenceTargets != null ? 1 : 0, rDim);
                }

                _mmtacOutputDHiddenKernel(new Index2D(sl, ed), bufDRegression.View, bufDRange.View, bufDQuality.View, bufDDirection.View, bufDMidDirection.View, bufDConfidence.View, bufRegressionProjection.View, bufRangeProjection.View, bufQualityProjection.View, bufDirectionProjection.View, bufMidDirectionProjection.View, bufConfidenceProjection.View, bufDHidden.View, useConfidence ? 1 : 0);

                _mmtacOutputProjectionGradKernel(new Index2D(rDim, ed), bufHidden.View, bufDRegression.View, bufRegressionProjectionGrad.View);
                _mmtacOutputBiasGradKernel(new Index1D(rDim), bufDRegression.View, bufRegressionBiasGrad.View);

                _mmtacOutputProjectionGradKernel(new Index2D(1, ed), bufHidden.View, bufDRange.View, bufRangeProjectionGrad.View);
                _mmtacOutputBiasGradKernel(new Index1D(1), bufDRange.View, bufRangeBiasGrad.View);

                _mmtacOutputProjectionGradKernel(new Index2D(1, ed), bufHidden.View, bufDQuality.View, bufQualityProjectionGrad.View);
                _mmtacOutputBiasGradKernel(new Index1D(1), bufDQuality.View, bufQualityBiasGrad.View);

                _mmtacOutputProjectionGradKernel(new Index2D(1, ed), bufHidden.View, bufDDirection.View, bufDirectionProjectionGrad.View);
                _mmtacOutputBiasGradKernel(new Index1D(1), bufDDirection.View, bufDirectionBiasGrad.View);

                _mmtacOutputProjectionGradKernel(new Index2D(1, ed), bufHidden.View, bufDMidDirection.View, bufMidDirectionProjectionGrad.View);
                _mmtacOutputBiasGradKernel(new Index1D(1), bufDMidDirection.View, bufMidDirectionBiasGrad.View);

                if (useConfidence)
                {
                    _mmtacOutputProjectionGradKernel(new Index2D(1, ed), bufHidden.View, bufDConfidence.View, bufConfidenceProjectionGrad.View);
                    _mmtacOutputBiasGradKernel(new Index1D(1), bufDConfidence.View, bufConfidenceBiasGrad.View);
                }

                var lossHost = new float[1];
                var dHidden = new float[sl, ed];
                bufLoss.CopyToCPU(lossHost);
                bufDHidden.CopyToCPU(dHidden);

                bufRegressionProjectionGrad.CopyToCPU(regressionProjectionGrad);
                bufRangeProjectionGrad.CopyToCPU(rangeProjectionGrad);
                bufQualityProjectionGrad.CopyToCPU(qualityProjectionGrad);
                bufDirectionProjectionGrad.CopyToCPU(directionProjectionGrad);
                bufMidDirectionProjectionGrad.CopyToCPU(midDirectionProjectionGrad);

                bufRegressionBiasGrad.CopyToCPU(regressionBiasGrad);
                bufRangeBiasGrad.CopyToCPU(rangeBiasGrad);
                bufQualityBiasGrad.CopyToCPU(qualityBiasGrad);
                bufDirectionBiasGrad.CopyToCPU(directionBiasGrad);
                bufMidDirectionBiasGrad.CopyToCPU(midDirectionBiasGrad);

                if (useConfidence)
                {
                    bufConfidenceProjectionGrad.CopyToCPU(confidenceProjectionGrad);
                    bufConfidenceBiasGrad.CopyToCPU(confidenceBiasGrad);
                }

                return (lossHost[0], dHidden);
            }
            finally
            {
                bufRegression.Dispose();
                bufRange.Dispose();
                bufQuality.Dispose();
                bufDirection.Dispose();
                bufMidDirection.Dispose();
                bufConfidence.Dispose();

                bufTargetRegression.Dispose();
                bufTargetRange.Dispose();
                bufTargetQuality.Dispose();
                bufTargetDirection.Dispose();
                bufTargetMidDirection.Dispose();

                bufPreviousClose.Dispose();
                bufConfidenceTargets.Dispose();
                bufHidden.Dispose();
                bufRegressionLogits.Dispose();
                bufRangeLogits.Dispose();

                bufRegressionProjection.Dispose();
                bufRangeProjection.Dispose();
                bufQualityProjection.Dispose();
                bufDirectionProjection.Dispose();
                bufMidDirectionProjection.Dispose();
                bufConfidenceProjection.Dispose();

                bufRegressionProjectionGrad.Dispose();
                bufRangeProjectionGrad.Dispose();
                bufQualityProjectionGrad.Dispose();
                bufDirectionProjectionGrad.Dispose();
                bufMidDirectionProjectionGrad.Dispose();
                bufConfidenceProjectionGrad.Dispose();

                bufRegressionBiasGrad.Dispose();
                bufRangeBiasGrad.Dispose();
                bufQualityBiasGrad.Dispose();
                bufDirectionBiasGrad.Dispose();
                bufMidDirectionBiasGrad.Dispose();
                bufConfidenceBiasGrad.Dispose();

                bufDRegression.Dispose();
                bufDRange.Dispose();
                bufDQuality.Dispose();
                bufDDirection.Dispose();
                bufDMidDirection.Dispose();
                bufDConfidence.Dispose();
                bufDHidden.Dispose();
                bufLoss.Dispose();
            }
        }

        public void AccumulateMmtacContextGradients(
                  float[,] dContextA,
                  float[,] dContextB,
                  float[,] contextTypeEmbeddingGrad,
                  float[,] dLiveNewsHidden,
                  float[] dGlobalHidden,
                  int numGlobal,
                  int numStoredNews,
                  int numNews,
                  int numLiveNews,
                  int numPriceContext,
                  int totalContext,
                  int priceOffset)
        {
            if (contextTypeEmbeddingGrad == null) throw new ArgumentNullException(nameof(contextTypeEmbeddingGrad));
            if (dContextA == null && dContextB == null) return;

            int ed = contextTypeEmbeddingGrad.GetLength(1);
            if (!ShouldUseGpu((long)Math.Max(1, totalContext) * ed))
            {
                _mutliThreadCPU.AccumulateMmtacContextGradients(dContextA, dContextB, contextTypeEmbeddingGrad, dLiveNewsHidden, dGlobalHidden, numGlobal, numStoredNews, numNews, numLiveNews, numPriceContext, totalContext, priceOffset);
                return;
            }

            int rowsA = dContextA != null ? dContextA.GetLength(0) : 1;
            int rowsB = dContextB != null ? dContextB.GetLength(0) : 1;
            var bufA = _accelerator.Allocate2DDenseX<float>(new Index2D(Math.Max(1, rowsA), ed));
            var bufB = _accelerator.Allocate2DDenseX<float>(new Index2D(Math.Max(1, rowsB), ed));
            var bufTypeGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(contextTypeEmbeddingGrad.GetLength(0), ed));
            var bufLive = _accelerator.Allocate2DDenseX<float>(new Index2D(Math.Max(1, numLiveNews), ed));
            var bufGlobal = _accelerator.Allocate1D<float>(ed);

            try
            {
                bufA.CopyFromCPU(dContextA ?? new float[Math.Max(1, rowsA), ed]);
                bufB.CopyFromCPU(dContextB ?? new float[Math.Max(1, rowsB), ed]);
                bufTypeGrad.CopyFromCPU(contextTypeEmbeddingGrad);
                bufLive.CopyFromCPU(dLiveNewsHidden ?? new float[Math.Max(1, numLiveNews), ed]);
                bufGlobal.CopyFromCPU(dGlobalHidden ?? new float[ed]);

                _accumulateMmtacContextTypeGradKernel(new Index2D(3, ed), bufA.View, bufB.View, bufTypeGrad.View, dContextA != null ? 1 : 0, dContextB != null ? 1 : 0, numGlobal, numNews, numPriceContext, totalContext, priceOffset);

                if (dLiveNewsHidden != null && numLiveNews > 0)
                {
                    _accumulateMmtacLiveNewsGradKernel(new Index2D(numLiveNews, ed), bufA.View, bufB.View, bufLive.View, dContextA != null ? 1 : 0, dContextB != null ? 1 : 0, numGlobal, numStoredNews, totalContext);
                }

                if (dGlobalHidden != null && numGlobal > 0)
                {
                    _accumulateMmtacGlobalGradKernel(new Index1D(ed), bufA.View, bufB.View, bufGlobal.View, dContextA != null ? 1 : 0, dContextB != null ? 1 : 0, numGlobal, totalContext);
                }

                bufTypeGrad.CopyToCPU(contextTypeEmbeddingGrad);
                if (dLiveNewsHidden != null && numLiveNews > 0)
                {
                    bufLive.CopyToCPU(dLiveNewsHidden);
                }

                if (dGlobalHidden != null && numGlobal > 0)
                {
                    bufGlobal.CopyToCPU(dGlobalHidden);
                }
            }
            finally
            {
                bufA.Dispose();
                bufB.Dispose();
                bufTypeGrad.Dispose();
                bufLive.Dispose();
                bufGlobal.Dispose();
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

            if (!ShouldUseGpu((long)ed * gd))
            {
                _mutliThreadCPU.AccumulateGlobalProjectionGradients(dGlobalHidden, globalFeatures, projectionGrad, biasGrad);
                return;
            }

            var bufDGlobal = _accelerator.Allocate1D<float>(ed);
            var bufFeatures = _accelerator.Allocate1D<float>(gd);
            var bufProjectionGrad = _accelerator.Allocate2DDenseX<float>(new Index2D(ed, gd));
            var bufBiasGrad = _accelerator.Allocate1D<float>(ed);

            try
            {
                bufDGlobal.CopyFromCPU(dGlobalHidden);
                bufFeatures.CopyFromCPU(globalFeatures);
                bufProjectionGrad.CopyFromCPU(projectionGrad);
                bufBiasGrad.CopyFromCPU(biasGrad);

                _accumulateGlobalProjectionGradKernel(new Index2D(ed, gd), bufDGlobal.View, bufFeatures.View, bufProjectionGrad.View, bufBiasGrad.View);

                bufProjectionGrad.CopyToCPU(projectionGrad);
                bufBiasGrad.CopyToCPU(biasGrad);
            }
            finally
            {
                bufDGlobal.Dispose();
                bufFeatures.Dispose();
                bufProjectionGrad.Dispose();
                bufBiasGrad.Dispose();
            }
        }
        public float[,] ExpandMeanPoolGradient(float[,] pooledGradient, int rowIndex, int rowCount, int embeddingDim)
        {
            if (pooledGradient == null) throw new ArgumentNullException(nameof(pooledGradient));
            if (rowCount <= 0) return new float[0, embeddingDim];
            if (rowIndex < 0 || rowIndex >= pooledGradient.GetLength(0)) throw new ArgumentOutOfRangeException(nameof(rowIndex));
            if (embeddingDim < 0 || embeddingDim > pooledGradient.GetLength(1)) throw new ArgumentOutOfRangeException(nameof(embeddingDim));

            if (!ShouldUseGpu((long)rowCount * embeddingDim))
            {
                return _mutliThreadCPU.ExpandMeanPoolGradient(pooledGradient, rowIndex, rowCount, embeddingDim);
            }

            var bufPooled = _accelerator.Allocate2DDenseX<float>(new Index2D(pooledGradient.GetLength(0), pooledGradient.GetLength(1)));
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(rowCount, embeddingDim));

            try
            {
                bufPooled.CopyFromCPU(pooledGradient);
                _expandMeanPoolGradientKernel(new Index2D(rowCount, embeddingDim), bufPooled.View, bufResult.View, rowIndex, rowCount);

                var result = new float[rowCount, embeddingDim];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufPooled.Dispose();
                bufResult.Dispose();
            }
        }


        private static float StableSigmoidDevice(float x)
        {
            if (x >= 0.0f)
            {
                float ex = XMath.Exp(-x);
                return 1.0f / (1.0f + ex);
            }

            float exp = XMath.Exp(x);
            return exp / (1.0f + exp);
        }

        private static float SoftplusDevice(float x)
        {
            if (x > 20.0f)
            {
                return x;
            }

            if (x < -20.0f)
            {
                return XMath.Exp(x);
            }

            return XMath.Log(1.0f + XMath.Exp(x));
        }

        private static float ClampDevice(float value, float min, float max)
        {
            if (value < min)
            {
                return min;
            }

            if (value > max)
            {
                return max;
            }

            return value;
        }

        private static float ReadContextGradDevice(ArrayView2D<float, Stride2D.DenseX> dContextA, ArrayView2D<float, Stride2D.DenseX> dContextB, int hasA, int hasB, int row, int col)
        {
            float g = 0.0f;

            if (hasA != 0 && row >= 0 && row < dContextA.Extent.X && col < dContextA.Extent.Y)
            {
                g += dContextA[row, col];
            }

            if (hasB != 0 && row >= 0 && row < dContextB.Extent.X && col < dContextB.Extent.Y)
            {
                g += dContextB[row, col];
            }

            return g;
        }
        private static void BuildMmtacContextWithPriceHiddenKernel(
            Index2D idx,
            ArrayView1D<float, Stride1D.Dense> globalToken,
            ArrayView2D<float, Stride2D.DenseX> newsHidden,
            ArrayView2D<float, Stride2D.DenseX> priceHidden,
            ArrayView2D<float, Stride2D.DenseX> contextTypeEmbedding,
            ArrayView2D<float, Stride2D.DenseX> output,
            int numGlobal,
            int numNews)
        {
            int row = idx.X;
            int d = idx.Y;

            if (row < numGlobal)
            {
                output[row, d] = globalToken[d] + contextTypeEmbedding[2, d];
                return;
            }

            int newsRow = row - numGlobal;
            if (newsRow < numNews)
            {
                output[row, d] = newsHidden[newsRow, d] + contextTypeEmbedding[0, d];
                return;
            }

            int priceRow = newsRow - numNews;
            output[row, d] = priceHidden[priceRow, d] + contextTypeEmbedding[1, d];
        }
        private static void BuildMmtacContextWithPriceTimesKernel(
            Index1D rowIndex,
            ArrayView1D<float, Stride1D.Dense> newsTimes,
            ArrayView1D<float, Stride1D.Dense> priceTimes,
            ArrayView1D<float, Stride1D.Dense> outputTimes,
            int numGlobal,
            int numNews,
            int hasNewsTimes,
            int hasPriceTimes)
        {
            int row = rowIndex;

            if (row < numGlobal)
            {
                outputTimes[row] = 0.0f;
                return;
            }

            int newsRow = row - numGlobal;
            if (newsRow < numNews)
            {
                outputTimes[row] = hasNewsTimes != 0 ? newsTimes[newsRow] : 0.0f;
                return;
            }

            int priceRow = newsRow - numNews;
            outputTimes[row] = hasPriceTimes != 0 ? priceTimes[priceRow] : 0.0f;
        }
        private static void AccumulateMmtacContextTypeGradKernel(
            Index2D idx,
            ArrayView2D<float, Stride2D.DenseX> dContextA,
            ArrayView2D<float, Stride2D.DenseX> dContextB,
            ArrayView2D<float, Stride2D.DenseX> contextTypeEmbeddingGrad,
            int hasA,
            int hasB,
            int numGlobal,
            int numNews,
            int numPriceContext,
            int totalContext,
            int priceOffset)
        {
            int type = idx.X;
            int d = idx.Y;
            float sum = 0.0f;

            if (type == 2)
            {
                for (int row = 0; row < numGlobal && row < totalContext; row++)
                {
                    sum += ReadContextGradDevice(dContextA, dContextB, hasA, hasB, row, d);
                }
            }
            else if (type == 0)
            {
                int start = numGlobal;
                int endCandidate = numGlobal + numNews;
                int end = endCandidate < totalContext ? endCandidate : totalContext;
                for (int row = start; row < end; row++)
                {
                    sum += ReadContextGradDevice(dContextA, dContextB, hasA, hasB, row, d);
                }
            }
            else
            {
                int start = priceOffset;
                int endCandidate = priceOffset + numPriceContext;
                int end = endCandidate < totalContext ? endCandidate : totalContext;
                for (int row = start; row < end; row++)
                {
                    sum += ReadContextGradDevice(dContextA, dContextB, hasA, hasB, row, d);
                }
            }

            contextTypeEmbeddingGrad[type, d] += sum;
        }

        private static void AccumulateMmtacLiveNewsGradKernel(
          Index2D idx,
          ArrayView2D<float, Stride2D.DenseX> dContextA,
          ArrayView2D<float, Stride2D.DenseX> dContextB,
          ArrayView2D<float, Stride2D.DenseX> dLiveNewsHidden,
          int hasA,
          int hasB,
          int numGlobal,
          int numStoredNews,
          int totalContext)
        {
            int liveIdx = idx.X;
            int d = idx.Y;
            int ctxIdx = numGlobal + numStoredNews + liveIdx;

            if (ctxIdx >= 0 && ctxIdx < totalContext)
            {
                dLiveNewsHidden[liveIdx, d] += ReadContextGradDevice(dContextA, dContextB, hasA, hasB, ctxIdx, d);
            }
        }

        private static void AccumulateMmtacGlobalGradKernel(
            Index1D dIndex,
            ArrayView2D<float, Stride2D.DenseX> dContextA,
            ArrayView2D<float, Stride2D.DenseX> dContextB,
            ArrayView1D<float, Stride1D.Dense> dGlobalHidden,
            int hasA,
            int hasB,
            int numGlobal,
            int totalContext)
        {
            int d = dIndex;
            float sum = 0.0f;

            for (int row = 0; row < numGlobal && row < totalContext; row++)
            {
                sum += ReadContextGradDevice(dContextA, dContextB, hasA, hasB, row, d);
            }

            dGlobalHidden[d] += sum;
        }

        private static void AccumulateGlobalProjectionGradKernel(
            Index2D idx,
            ArrayView1D<float, Stride1D.Dense> dGlobalHidden,
            ArrayView1D<float, Stride1D.Dense> globalFeatures,
            ArrayView2D<float, Stride2D.DenseX> projectionGrad,
            ArrayView1D<float, Stride1D.Dense> biasGrad)
        {
            int d = idx.X;
            int g = idx.Y;
            float tokenGrad = dGlobalHidden[d];

            projectionGrad[d, g] += tokenGrad * globalFeatures[g];

            if (g == 0)
            {
                biasGrad[d] += tokenGrad;
            }
        }

        private static void ExpandMeanPoolGradientKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> pooledGradient, ArrayView2D<float, Stride2D.DenseX> result, int rowIndex, int rowCount)
        {
            int t = idx.X;
            int d = idx.Y;
            result[t, d] = pooledGradient[rowIndex, d] / rowCount;
        }

        private static void MmtacRegressionBaseGradKernel(
            Index1D rowIndex,
            ArrayView2D<float, Stride2D.DenseX> regression,
            ArrayView2D<float, Stride2D.DenseX> targetRegression,
            ArrayView2D<float, Stride2D.DenseX> targetDirection,
            ArrayView1D<float, Stride1D.Dense> previousClose,
            ArrayView2D<float, Stride2D.DenseX> dRegression,
            ArrayView1D<float, Stride1D.Dense> loss,
            float closeDirectionConsistencyWeight,
            float closeDirectionConsistencyMargin,
            int hasPreviousClose)
        {
            int t = rowIndex;
            int sl = (int)regression.Extent.X;
            int rDim = (int)regression.Extent.Y;

            float invRegCount = 1.0f / (sl * rDim);
            float invSl = 1.0f / sl;

            float diffHigh = regression[t, 0] - targetRegression[t, 0];
            float diffLow = regression[t, 1] - targetRegression[t, 1];
            float diffClose = regression[t, 2] - targetRegression[t, 2];

            Atomic.Add(
                ref loss[0],
                (diffHigh * diffHigh + diffLow * diffLow + diffClose * diffClose) * invRegCount);

            float dHigh = 2.0f * diffHigh * invRegCount;
            float dLow = 2.0f * diffLow * invRegCount;
            float dClose = 2.0f * diffClose * invRegCount;

            if (closeDirectionConsistencyWeight > 0.0f && hasPreviousClose != 0)
            {
                float sign = targetDirection[t, 0] >= 0.5f ? 1.0f : -1.0f;

                float z = sign *
                    (regression[t, 2] - previousClose[t] - sign * closeDirectionConsistencyMargin);

                float closePenalty;

                if (z > 20.0f)
                {
                    closePenalty = XMath.Exp(-z);
                }
                else if (z < -20.0f)
                {
                    closePenalty = -z;
                }
                else
                {
                    closePenalty = XMath.Log(1.0f + XMath.Exp(-z));
                }

                Atomic.Add(
                    ref loss[0],
                    closeDirectionConsistencyWeight * closePenalty * invSl);

                float sigmoidNegZ;

                if (z >= 0.0f)
                {
                    float ez = XMath.Exp(-z);
                    sigmoidNegZ = ez / (1.0f + ez);
                }
                else
                {
                    float ez = XMath.Exp(z);
                    sigmoidNegZ = 1.0f / (1.0f + ez);
                }

                dClose += -sign * sigmoidNegZ * closeDirectionConsistencyWeight * invSl;
            }

            // Temporary storage:
            // dRegression[t,0] = dHigh
            // dRegression[t,1] = dLow
            // dRegression[t,2] = dClose
            // MmtacRangeShareGradKernel converts these into raw regression-logit gradients.
            dRegression[t, 0] = dHigh;
            dRegression[t, 1] = dLow;
            dRegression[t, 2] = dClose;
        }

        private static void MmtacRangeShareGradKernel(
            Index1D rowIndex,
            ArrayView2D<float, Stride2D.DenseX> range,
            ArrayView2D<float, Stride2D.DenseX> targetRange,
            ArrayView2D<float, Stride2D.DenseX> regressionLogits,
            ArrayView1D<float, Stride1D.Dense> rangeLogits,
            ArrayView2D<float, Stride2D.DenseX> dRegression,
            ArrayView2D<float, Stride2D.DenseX> dRangeLogit,
            ArrayView1D<float, Stride1D.Dense> loss,
            float rangeLossWeight)
        {
            int t = rowIndex;
            int sl = (int)range.Extent.X;
            float invSl = 1.0f / sl;

            float dHigh = dRegression[t, 0];
            float dLow = dRegression[t, 1];
            float dClose = dRegression[t, 2];

            float rangeDiff = range[t, 0] - targetRange[t, 0];

            Atomic.Add(
                ref loss[0],
                rangeLossWeight * rangeDiff * rangeDiff * invSl);

            float dRangeOutput = 2.0f * rangeDiff * invSl * rangeLossWeight;

            float upLogit = regressionLogits[t, 0];
            float downLogit = regressionLogits[t, 1];
            float rangeLogit = rangeLogits[t];

            float upBase = SoftplusDevice(upLogit);
            float downBase = SoftplusDevice(downLogit);
            float den = upBase + downBase;

            float upShare = den > 1e-6f ? upBase / den : 0.5f;
            float downShare = 1.0f - upShare;
            float rangeValue = SoftplusDevice(rangeLogit);

            float dCloseRaw = dHigh + dLow + dClose;
            float dRangeValue = dHigh * upShare - dLow * downShare + dRangeOutput;
            float dShare = rangeValue * (dHigh + dLow);

            float dUpBase = 0.0f;
            float dDownBase = 0.0f;

            if (den > 1e-6f)
            {
                float invDenSq = 1.0f / (den * den);
                dUpBase = dShare * downBase * invDenSq;
                dDownBase = -dShare * upBase * invDenSq;
            }

            dRegression[t, 0] = dUpBase * StableSigmoidDevice(upLogit);
            dRegression[t, 1] = dDownBase * StableSigmoidDevice(downLogit);
            dRegression[t, 2] = dCloseRaw;

            dRangeLogit[t, 0] = dRangeValue * StableSigmoidDevice(rangeLogit);
        }

        private static void MmtacScalarMseGradKernel(
            Index1D rowIndex,
            ArrayView2D<float, Stride2D.DenseX> prediction,
            ArrayView2D<float, Stride2D.DenseX> target,
            ArrayView2D<float, Stride2D.DenseX> dLogit,
            ArrayView1D<float, Stride1D.Dense> loss,
            float weight)
        {
            int t = rowIndex;
            int sl = (int)prediction.Extent.X;
            float p = prediction[t, 0];
            float diff = p - target[t, 0];
            float invSl = 1.0f / sl;

            Atomic.Add(ref loss[0], weight * diff * diff * invSl);
            dLogit[t, 0] = 2.0f * diff * invSl * weight * p * (1.0f - p);
        }

        private static void MmtacScalarBceGradKernel(
            Index1D rowIndex,
            ArrayView2D<float, Stride2D.DenseX> prediction,
            ArrayView2D<float, Stride2D.DenseX> target,
            ArrayView2D<float, Stride2D.DenseX> dLogit,
            ArrayView1D<float, Stride1D.Dense> loss,
            float weight)
        {
            int t = rowIndex;
            int sl = (int)prediction.Extent.X;
            float p = prediction[t, 0];
            float y = target[t, 0];
            float pc = ClampDevice(p, 1e-7f, 1.0f - 1e-7f);
            float invSl = 1.0f / sl;

            Atomic.Add(ref loss[0], weight * (-(y * XMath.Log(pc) + (1.0f - y) * XMath.Log(1.0f - pc))) * invSl);
            dLogit[t, 0] = (p - y) * weight * invSl;
        }

        private static void MmtacConfidenceGradKernel(
            Index1D rowIndex,
            ArrayView2D<float, Stride2D.DenseX> confidence,
            ArrayView1D<float, Stride1D.Dense> confidenceTargets,
            ArrayView2D<float, Stride2D.DenseX> regression,
            ArrayView2D<float, Stride2D.DenseX> targetRegression,
            ArrayView2D<float, Stride2D.DenseX> dConfidence,
            ArrayView1D<float, Stride1D.Dense> loss,
            float weight,
            int hasConfidenceTargets,
            int rDim)
        {
            int t = rowIndex;
            int sl = (int)confidence.Extent.X;
            float p = confidence[t, 0];
            float y;

            if (hasConfidenceTargets != 0)
            {
                y = confidenceTargets[t];
            }
            else
            {
                float sq = 0.0f;
                for (int j = 0; j < rDim; j++)
                {
                    float diff = regression[t, j] - targetRegression[t, j];
                    sq += diff * diff;
                }

                y = XMath.Exp(-5.0f * XMath.Sqrt(sq / rDim));
            }

            float pc = ClampDevice(p, 1e-7f, 1.0f - 1e-7f);
            float invSl = 1.0f / sl;

            Atomic.Add(ref loss[0], weight * (-(y * XMath.Log(pc) + (1.0f - y) * XMath.Log(1.0f - pc))) * invSl);
            dConfidence[t, 0] = (p - y) * weight * invSl;
        }

        private static void MmtacOutputDHiddenKernel(
            Index2D idx,
            ArrayView2D<float, Stride2D.DenseX> dRegression,
            ArrayView2D<float, Stride2D.DenseX> dRange,
            ArrayView2D<float, Stride2D.DenseX> dQuality,
            ArrayView2D<float, Stride2D.DenseX> dDirection,
            ArrayView2D<float, Stride2D.DenseX> dMidDirection,
            ArrayView2D<float, Stride2D.DenseX> dConfidence,
            ArrayView2D<float, Stride2D.DenseX> regressionProjection,
            ArrayView2D<float, Stride2D.DenseX> rangeProjection,
            ArrayView2D<float, Stride2D.DenseX> qualityProjection,
            ArrayView2D<float, Stride2D.DenseX> directionProjection,
            ArrayView2D<float, Stride2D.DenseX> midDirectionProjection,
            ArrayView2D<float, Stride2D.DenseX> confidenceProjection,
            ArrayView2D<float, Stride2D.DenseX> dHidden,
            int useConfidence)
        {
            int t = idx.X;
            int k = idx.Y;
            int rDim = (int)dRegression.Extent.Y;
            float sum = 0.0f;

            for (int v = 0; v < rDim; v++)
            {
                sum += dRegression[t, v] * regressionProjection[v, k];
            }

            sum += dRange[t, 0] * rangeProjection[0, k];
            sum += dQuality[t, 0] * qualityProjection[0, k];
            sum += dDirection[t, 0] * directionProjection[0, k];
            sum += dMidDirection[t, 0] * midDirectionProjection[0, k];

            if (useConfidence != 0)
            {
                sum += dConfidence[t, 0] * confidenceProjection[0, k];
            }

            dHidden[t, k] = sum;
        }

        private static void MmtacOutputProjectionGradKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> hidden, ArrayView2D<float, Stride2D.DenseX> dLogits, ArrayView2D<float, Stride2D.DenseX> projectionGrad)
        {
            int o = idx.X;
            int k = idx.Y;
            int rows = (int)hidden.Extent.X;
            float sum = 0.0f;

            for (int t = 0; t < rows; t++)
            {
                sum += dLogits[t, o] * hidden[t, k];
            }

            projectionGrad[o, k] += sum;
        }

        private static void MmtacOutputBiasGradKernel(Index1D oIndex, ArrayView2D<float, Stride2D.DenseX> dLogits, ArrayView1D<float, Stride1D.Dense> biasGrad)
        {
            int o = oIndex;
            int rows = (int)dLogits.Extent.X;
            float sum = 0.0f;

            for (int t = 0; t < rows; t++)
            {
                sum += dLogits[t, o];
            }

            biasGrad[o] += sum;
        }
    }
}
