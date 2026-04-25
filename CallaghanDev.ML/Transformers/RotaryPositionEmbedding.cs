using System;

namespace CallaghanDev.ML.Transformers
{
    public static class RotaryPositionEmbedding
    {
        private const float BaseTheta = 10000f;

        public static void ApplyInPlace(float[,] q, float[,] k, int numHeads)
        {
            if (q == null)
            {
                throw new ArgumentNullException(nameof(q));
            }
            if (k == null)
            {
                throw new ArgumentNullException(nameof(k));
            }

            int embeddingDim = q.GetLength(1);

            if (k.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("Q and K must have the same embedding dimension.");
            }

            int headDim = embeddingDim / numHeads;
            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by number of heads.");
            }
            if ((headDim & 1) != 0)
            {
                throw new ArgumentException("RoPE requires an even per-head dimension.");
            }

            ApplyInPlace(q, numHeads);
            ApplyInPlace(k, numHeads);
        }

        public static void ApplyInPlace(float[,] x, int numHeads)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));

            int embeddingDim = x.GetLength(1);
            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by number of heads.");
            }
            if ((headDim & 1) != 0)
            {
                throw new ArgumentException("RoPE requires an even per-head dimension.");
            }

            for (int head = 0; head < numHeads; head++)
            {
                int start = head * headDim;
                RotateHeadInPlace(x, start, headDim, inverse: false);
            }
        }

        public static void ApplyBackwardInPlace(float[,] dQ, float[,] dK, int numHeads)
        {
            if (dQ == null)
            {
                throw new ArgumentNullException(nameof(dQ));
            }
            if (dK == null)
            {
                throw new ArgumentNullException(nameof(dK));
            }

            int embeddingDim = dQ.GetLength(1);
            if (dK.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("dQ and dK must have the same embedding dimension.");
            }

            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by number of heads.");
            }
            if ((headDim & 1) != 0)
            {
                throw new ArgumentException("RoPE requires an even per-head dimension.");
            }

            ApplyBackwardInPlace(dQ, numHeads);
            ApplyBackwardInPlace(dK, numHeads);
        }

        public static void ApplyBackwardInPlace(float[,] dX, int numHeads)
        {
            if (dX == null)
            {
                throw new ArgumentNullException(nameof(dX));
            }

            int embeddingDim = dX.GetLength(1);
            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by number of heads.");
            }
            if ((headDim & 1) != 0)
            {
                throw new ArgumentException("RoPE requires an even per-head dimension.");
            }

            for (int head = 0; head < numHeads; head++)
            {
                int start = head * headDim;
                RotateHeadInPlace(dX, start, headDim, inverse: true);
            }
        }

        private static void RotateHeadInPlace(float[,] matrix, int startCol, int headDim, bool inverse)
        {
            int seqLen = matrix.GetLength(0);
            int pairCount = headDim / 2;

            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int pair = 0; pair < pairCount; pair++)
                {
                    int evenCol = startCol + (pair * 2);
                    int oddCol = evenCol + 1;

                    float theta = pos / MathF.Pow(BaseTheta, (2f * pair) / headDim);
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
    }
}