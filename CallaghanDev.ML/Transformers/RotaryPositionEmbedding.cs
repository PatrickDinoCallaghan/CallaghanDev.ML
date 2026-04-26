using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.AccelerationManagers.GPU;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
using System;

namespace CallaghanDev.ML.Transformers
{
    public class RotaryPositionEmbedding
    {
        private const float BaseTheta = 10000f;
        private readonly IAccelerationManager _accel;

        public RotaryPositionEmbedding(RuntimeConfig runtime) : this(AccelerationFactory.Create(runtime))
        {

        }

        public RotaryPositionEmbedding(IAccelerationManager accelerationManager)
        {
            _accel = accelerationManager;
        }


        public void ApplyInPlace(float[,] q, float[,] k, int numHeads)
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

        public void ApplyInPlace(float[,] x, int numHeads)
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

            _accel.ApplyRotaryPositionEmbeddingInPlace(
                x,
                numHeads,
                BaseTheta,
                inverse: false);
        }

        public void ApplyBackwardInPlace(float[,] dQ, float[,] dK, int numHeads)
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

        public void ApplyBackwardInPlace(float[,] dX, int numHeads)
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
            _accel.ApplyRotaryPositionEmbeddingInPlace(dX, numHeads, BaseTheta, inverse: true);
       
        }
    }
}