using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MultiTypeTransformer
{
    public abstract class TransformerBlockBase
    {
        protected readonly int EmbeddingDim;
        protected readonly int NumHeads;
        protected readonly int HeadDim;
        protected readonly IAccelerationManager Accel;

        public MultiHeadAttention SelfAttention;
        public NeuralNetwork FeedForwardNetwork;

        public float[] LNSelfGamma;
        public float[] LNSelfBeta;

        public float[] LNFFNGamma;
        public float[] LNFFNBeta;

        protected TransformerBlockBase(int embeddingDim, int numHeads, IAccelerationManager accel)
        {
            EmbeddingDim = embeddingDim;
            NumHeads = numHeads;
            HeadDim = embeddingDim / numHeads;
            Accel = accel;
        }

        public float[,] Forward(float[,] x, bool[,] selfMask = null)
        {
            var selfOut = SelfAttentionForward(x, selfMask);
            var normedSelf = ApplyResidualNorm(x, selfOut, LNSelfGamma, LNSelfBeta);

            var afterCross = CrossAttentionForward(normedSelf);

            var ffOut = FeedForward(afterCross);
            return ApplyResidualNorm(afterCross, ffOut, LNFFNGamma, LNFFNBeta);
        }

        protected virtual float[,] SelfAttentionForward(float[,] x, bool[,] mask)
        {
            return SelfAttention.Forward(x, mask);
        }

        protected abstract float[,] CrossAttentionForward(float[,] x);

        protected virtual float[,] FeedForward(float[,] x)
        {
            int seqLen = x.GetLength(0);
            return Accel.FFNForwardBatch(x, seqLen, EmbeddingDim, FeedForwardNetwork.ForwardPassOnly);
        }

        protected float[,] ApplyResidualNorm(float[,] input, float[,] subLayer, float[] gamma, float[] beta)
        {
            var residual = Accel.MatrixAdd(input, subLayer);
            return Accel.LayerNorm(residual, gamma, beta);
        }
    }

    public class TransformerBlock : TransformerBlockBase
    {
        public MultiHeadAttention Attention { get; set; }
        public new NeuralNetwork FeedForwardNetwork { get; set; }

        public float[] LN1Gamma { get; set; }
        public float[] LN1Beta { get; set; }
        public float[] LN2Gamma { get; set; }
        public float[] LN2Beta { get; set; }

        /// <summary>
        /// Explicit-parameter constructor. Takes only what TransformerBlock
        /// actually uses — no throwaway TransformerConfig needed at call sites.
        /// </summary>
        public TransformerBlock(
            int embeddingDim,
            int numHeads,
            int feedForwardDim,
            ActivationType ffnActivationType,
            CostFunctionType costFunction,
            ActivationDistribution activationDistribution,
            float l2RegulationLamda,
            float gradientClippingThreshold,
            AccelerationType accelerationType,
            int accelerationDeviceId,
            IAccelerationManager accel,
            Random random = null)
            : base(embeddingDim, numHeads, accel)
        {
            if (embeddingDim % numHeads != 0)
                throw new InvalidOperationException(
                    $"EmbeddingDim ({embeddingDim}) must be divisible by NumHeads ({numHeads})");

            SelfAttention = new MultiHeadAttention(embeddingDim, numHeads, accel, random);

            LNSelfGamma = Enumerable.Repeat(1.0f, embeddingDim).ToArray();
            LNSelfBeta = new float[embeddingDim];

            LNFFNGamma = Enumerable.Repeat(1.0f, embeddingDim).ToArray();
            LNFFNBeta = new float[embeddingDim];

            // Keep LN1/LN2 aliases pointing at the same arrays so any code
            // that reads block.LN1Gamma still works unchanged.
            LN1Gamma = LNSelfGamma;
            LN1Beta = LNSelfBeta;
            LN2Gamma = LNFFNGamma;
            LN2Beta = LNFFNBeta;

            var ffnParams = new Parameters
            {
                LayerWidths = new List<int>
                {
                    embeddingDim,
                    feedForwardDim,
                    embeddingDim
                },
                LayerActivations = new List<ActivationType>
                {
                    ActivationType.None,
                    ffnActivationType,
                    ActivationType.None
                },
                AccelerationType = accelerationType,
                AccelerationDeviceId = accelerationDeviceId,
                CostFunction = costFunction,
                ActivationDistribution = activationDistribution,
                L2RegulationLamda = l2RegulationLamda,
                GradientClippingThreshold = gradientClippingThreshold,
                inputActivationMin = Enumerable.Repeat(0f, embeddingDim).ToArray(),
                inputActivationMax = Enumerable.Repeat(1f, embeddingDim).ToArray()
            };

            FeedForwardNetwork = new NeuralNetwork(ffnParams);
            base.FeedForwardNetwork = FeedForwardNetwork;

            Attention = SelfAttention;
        }

        protected override float[,] CrossAttentionForward(float[,] x)
        {
            // Plain TransformerBlock is self-attention only.
            // There is no external context, so just pass through unchanged.
            return x;
        }
        public void ReplaceFeedForwardNetwork(NeuralNetwork network)
        {
            if (network == null)
            {
                throw new ArgumentNullException(nameof(network));
            }

            FeedForwardNetwork = network;
            base.FeedForwardNetwork = network;
        }
    }
}
