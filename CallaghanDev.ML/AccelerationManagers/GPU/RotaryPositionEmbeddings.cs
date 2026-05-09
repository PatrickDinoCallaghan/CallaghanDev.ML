using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, int, int, float, int> _rotaryHeadKernel;
        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, int, int, float, int> _rotaryAllHeadsKernel;

        private void InitRotaryPositionEmbeddingKernels()
        {
            _rotaryHeadKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, int, int, float, int>(RotaryHeadKernel);
            _rotaryAllHeadsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, int, int, float, int>(RotaryAllHeadsKernel);
        }

        private void DisposeRotaryPositionEmbeddingBuffers()
        {
        }

        #region Rotary Position Embeddings

        public void ApplyRotaryPositionEmbeddingHeadInPlace(float[,] matrix, int startCol, int headDim, float baseTheta, bool inverse)
        {
            ValidateRotaryHead(matrix, startCol, headDim, baseTheta);

            int seqLen = matrix.GetLength(0);
            int embeddingDim = matrix.GetLength(1);
            int pairCount = headDim / 2;

            if (pairCount == 0)
            {
                return;
            }

            if (!ShouldUseGpu((long)seqLen * headDim))
            {
                ApplyRotaryPositionEmbeddingHeadCoreInPlace(matrix, startCol, headDim, baseTheta, inverse);
                return;
            }

            var bufMatrix = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));

            try
            {
                bufMatrix.CopyFromCPU(matrix);
                _rotaryHeadKernel(new Index2D(seqLen, pairCount), bufMatrix.View, startCol, headDim, baseTheta, inverse ? 1 : 0);
                bufMatrix.CopyToCPU(matrix);
            }
            finally
            {
                bufMatrix.Dispose();
            }
        }

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

            ValidateRotaryBaseTheta(baseTheta);

            int seqLen = matrix.GetLength(0);
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

            if (headDim == 0)
            {
                return;
            }

            if (!ShouldUseGpu((long)seqLen * embeddingDim))
            {
                for (int head = 0; head < numHeads; head++)
                {
                    ApplyRotaryPositionEmbeddingHeadCoreInPlace(matrix, head * headDim, headDim, baseTheta, inverse);
                }

                return;
            }

            int totalPairs = embeddingDim / 2;
            var bufMatrix = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));

            try
            {
                bufMatrix.CopyFromCPU(matrix);
                _rotaryAllHeadsKernel(new Index2D(seqLen, totalPairs), bufMatrix.View, numHeads, headDim, baseTheta, inverse ? 1 : 0);
                bufMatrix.CopyToCPU(matrix);
            }
            finally
            {
                bufMatrix.Dispose();
            }
        }

        private static void RotaryHeadKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> matrix, int startCol, int headDim, float baseTheta, int inverse)
        {
            int pos = idx.X;
            int pair = idx.Y;
            int evenCol = startCol + (pair * 2);
            int oddCol = evenCol + 1;

            ApplyRotaryPair(matrix, pos, evenCol, oddCol, pair, headDim, baseTheta, inverse);
        }

        private static void RotaryAllHeadsKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> matrix, int numHeads, int headDim, float baseTheta, int inverse)
        {
            int pos = idx.X;
            int globalPair = idx.Y;
            int pairsPerHead = headDim / 2;
            int head = globalPair / pairsPerHead;
            int pair = globalPair - (head * pairsPerHead);
            int evenCol = (head * headDim) + (pair * 2);
            int oddCol = evenCol + 1;

            ApplyRotaryPair(matrix, pos, evenCol, oddCol, pair, headDim, baseTheta, inverse);
        }

        private static void ApplyRotaryPair(ArrayView2D<float, Stride2D.DenseX> matrix, int pos, int evenCol, int oddCol, int pair, int headDim, float baseTheta, int inverse)
        {
            float theta = pos / XMath.Pow(baseTheta, (2.0f * pair) / headDim);
            float cos = XMath.Cos(theta);
            float sin = XMath.Sin(theta);

            if (inverse != 0)
            {
                sin = -sin;
            }

            float x0 = matrix[pos, evenCol];
            float x1 = matrix[pos, oddCol];

            matrix[pos, evenCol] = (x0 * cos) - (x1 * sin);
            matrix[pos, oddCol] = (x0 * sin) + (x1 * cos);
        }

        private static void ValidateRotaryHead(float[,] matrix, int startCol, int headDim, float baseTheta)
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

            ValidateRotaryBaseTheta(baseTheta);

            int embeddingDim = matrix.GetLength(1);
            if (startCol + headDim > embeddingDim)
            {
                throw new ArgumentException("Head range exceeds matrix embedding dimension.");
            }
        }

        private static void ValidateRotaryBaseTheta(float baseTheta)
        {
            if (baseTheta <= 0.0f || float.IsNaN(baseTheta) || float.IsInfinity(baseTheta))
            {
                throw new ArgumentOutOfRangeException(nameof(baseTheta));
            }
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
                    float theta = pos / MathF.Pow(baseTheta, (2.0f * pair) / headDim);
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
    }
}
