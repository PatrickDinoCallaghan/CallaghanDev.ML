using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.AccelerationManagers.GPU;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using ILGPU.Runtime;
using System;

namespace CallaghanDev.ML.Transformers.CrossAttentionMultimodal
{
    public class TransformerBlock
    {
        public MultiHeadAttention SelfAttention { get; set; }
        public float[] LNSelfGamma { get; set; }
        public float[] LNSelfBeta { get; set; }

        public MultiHeadAttention CrossAttention { get; set; }
        public float[] LNCrossGamma { get; set; }
        public float[] LNCrossBeta { get; set; }

        public NeuralNetwork FeedForwardNetwork { get; set; }
        public float[] LNFFNGamma { get; set; }
        public float[] LNFFNBeta { get; set; }

        private readonly int _embeddingDim;

        private readonly int _numHeads;

        private readonly RotaryPositionEmbedding _rotaryPositionEmbedding;

        private readonly IAccelerationManager _accel;

        public TransformerBlock(int embeddingDim, int numHeads, int feedForwardDim, ActivationType ffnActivation, Random random, AccelerationType accelerationType, int AccelerationDeviceId = 0, float l2Lambda = 0.01f)
        {
            if (embeddingDim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(embeddingDim));
            }
            if (numHeads <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numHeads));
            }
            if (feedForwardDim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(feedForwardDim));
            }

            if (embeddingDim % numHeads != 0)
            {
                throw new InvalidOperationException($"EmbeddingDim ({embeddingDim}) must be divisible by NumHeads ({numHeads})");
            }

            if (accelerationType == AccelerationType.GPU || accelerationType == AccelerationType.CUDA)
            {
                _accel = new AccelerationGPU(accelerationType, AccelerationDeviceId);
            }
            else if (accelerationType == AccelerationType.CPU)
            {
                _accel = new AccelerationCPU();
            }
            else if (accelerationType == AccelerationType.MultiThreadCPU)
            {
                _accel = new AccelerationMutliThreadCPU();
            }
            else
            {
                throw new Exception("Unsupported AccelerationType");
            }

            random ??= new Random();

            _numHeads = numHeads;
            _embeddingDim = embeddingDim;

            SelfAttention = new MultiHeadAttention(embeddingDim, _numHeads, _accel, random);
            LNSelfGamma = new float[embeddingDim];
            LNSelfBeta = new float[embeddingDim];

            CrossAttention = new MultiHeadAttention(embeddingDim, _numHeads, _accel, random);
            LNCrossGamma = new float[embeddingDim];
            LNCrossBeta = new float[embeddingDim];

            var parameters = new Parameters
            {
                AccelerationType = accelerationType,
                AccelerationDeviceId = AccelerationDeviceId,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Normal,
                LayerWidths = new List<int> { embeddingDim, feedForwardDim, embeddingDim },

                // Keep the FFN output projection linear. The previous code used ffnActivation
                // for the output layer too, which is especially harmful with Relu.
                LayerActivations = new List<ActivationType> { ffnActivation, ffnActivation, ActivationType.None },
                L2RegulationLamda = l2Lambda
            };

            FeedForwardNetwork = new NeuralNetwork(parameters);

            LNFFNGamma = new float[embeddingDim];
            LNFFNBeta = new float[embeddingDim];

            for (int i = 0; i < embeddingDim; i++)
            {
                LNSelfGamma[i] = 1.0f;
                LNCrossGamma[i] = 1.0f;
                LNFFNGamma[i] = 1.0f;
            }
        }
        private void Init()
        {

        }
        public float[,] Forward(float[,] x, float[,] textHidden, bool[,] selfAttnMask, IAccelerationManager accel)
        {
            int priceSeqLen = x.GetLength(0);
            int numHeads = _numHeads;
            int headDim = _embeddingDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Self-attention
            var selfQ = ComputeProjection(x, SelfAttention.WQ, SelfAttention.BiasQ, accel);
            var selfK = ComputeProjection(x, SelfAttention.WK, SelfAttention.BiasK, accel);
            var selfV = ComputeProjection(x, SelfAttention.WV, SelfAttention.BiasV, accel);

            _rotaryPositionEmbedding.ApplyInPlace(selfQ, selfK, numHeads);

            var selfAttnOut = accel.MultiHeadAttentionForward(selfQ, selfK, selfV, numHeads, scale, selfAttnMask);
            var selfProjected = ComputeProjection(selfAttnOut, SelfAttention.WO, SelfAttention.BiasO, accel);

            var selfResidual = accel.MatrixAdd(x, selfProjected);
            var (normedSelf, _, _, _) = accel.LayerNormForward(selfResidual, LNSelfGamma, LNSelfBeta);

            float[,] normedCross;
            if (textHidden != null)
            {
                var crossQ = ComputeProjection(normedSelf, CrossAttention.WQ, CrossAttention.BiasQ, accel);
                var crossK = ComputeProjection(textHidden, CrossAttention.WK, CrossAttention.BiasK, accel);
                var crossV = ComputeProjection(textHidden, CrossAttention.WV, CrossAttention.BiasV, accel);

                _rotaryPositionEmbedding.ApplyInPlace(crossQ, crossK, numHeads);

                var crossAttnOut = accel.MultiHeadAttentionForward(crossQ, crossK, crossV, numHeads, scale, null);
                var crossProjected = ComputeProjection(crossAttnOut, CrossAttention.WO, CrossAttention.BiasO, accel);

                var crossResidual = accel.MatrixAdd(normedSelf, crossProjected);
                var (normedCrossResult, _, _, _) = accel.LayerNormForward(crossResidual, LNCrossGamma, LNCrossBeta);
                normedCross = normedCrossResult;
            }
            else
            {
                var (normedCrossResult, _, _, _) = accel.LayerNormForward(normedSelf, LNCrossGamma, LNCrossBeta);
                normedCross = normedCrossResult;
            }

            var ffOutput = new float[priceSeqLen, _embeddingDim];
            for (int i = 0; i < priceSeqLen; i++)
            {
                var inputRow = accel.ExtractRow(normedCross, i, _embeddingDim);
                var outputRow = FeedForwardNetwork.ForwardPassOnly(inputRow);
                accel.SetRow(ffOutput, i, outputRow, _embeddingDim);
            }

            var ffResidual = accel.MatrixAdd(normedCross, ffOutput);
            var (normedFF, _, _, _) = accel.LayerNormForward(ffResidual, LNFFNGamma, LNFFNBeta);

            return normedFF;
        }
        private float[,] ComputeProjection(float[,] input, float[,] weight, float[] bias, IAccelerationManager accel)
        {
            /*
            var projected = accel.BatchDotProduct(weight, input);
            int rows = projected.GetLength(0);
            int cols = projected.GetLength(1);

            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = projected[i, j] + bias[j];
                }
            }
            return result;*/


            var projected = accel.BatchDotProduct(weight, input);

            return accel.MatrixAddBias(projected, bias);
        }
    }

    public class CrossAttentionBlock : TransformerBlockBase
    {
        private readonly MultiHeadAttention _crossAttention;

        private readonly float[] _lnCrossGamma;
        private readonly float[] _lnCrossBeta;

        private float[,] _context;

        public CrossAttentionBlock(
            int embeddingDim,
            int numHeads,
            int feedForwardDim,
            ActivationType ffnActivation,
            IAccelerationManager accel,
            Random random,
            AccelerationType accelType = AccelerationType.CPU,
            int accelDeviceId = 0,
            float l2Lambda = 0.01f)
            : base(embeddingDim, numHeads, accel)
        {
            if (embeddingDim % numHeads != 0)
                throw new InvalidOperationException($"EmbeddingDim ({embeddingDim}) must be divisible by NumHeads ({numHeads})");

            // ---- Self Attention (base uses this) ----
            SelfAttention = new MultiHeadAttention(embeddingDim, numHeads, accel, random);

            LNSelfGamma = new float[embeddingDim];
            LNSelfBeta = new float[embeddingDim];

            // ---- Cross Attention ----
            _crossAttention = new MultiHeadAttention(embeddingDim, numHeads, accel, random);

            _lnCrossGamma = new float[embeddingDim];
            _lnCrossBeta = new float[embeddingDim];

            // ---- Feed Forward ----
            var parameters = new Parameters
            {
                AccelerationType = accelType,
                AccelerationDeviceId = accelDeviceId,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Normal,
                LayerWidths = new List<int>
                {
                    embeddingDim,
                    feedForwardDim,
                    embeddingDim
                },
                LayerActivations = new List<ActivationType>
                {
                    ffnActivation,
                    ffnActivation,
                    ActivationType.None
                },
                L2RegulationLamda = l2Lambda
            };

            FeedForwardNetwork = new NeuralNetwork(parameters);

            LNFFNGamma = new float[embeddingDim];
            LNFFNBeta = new float[embeddingDim];

            // ---- Initialise LayerNorm gammas to 1 ----
            for (int i = 0; i < embeddingDim; i++)
            {
                LNSelfGamma[i] = 1.0f;
                _lnCrossGamma[i] = 1.0f;
                LNFFNGamma[i] = 1.0f;
            }
        }

        /// <summary>
        /// Set external context (e.g. encoder output, text embeddings, memory, etc.)
        /// </summary>
        public void SetContext(float[,] context)
        {
            _context = context;
        }

        /// <summary>
        /// Cross-attention stage (overrides base hook)
        /// </summary>
        protected override float[,] CrossAttentionForward(float[,] x)
        {
            // No context → just normalise (keeps distribution stable for FFN)
            if (_context == null)
            {
                return Accel.LayerNorm(x, _lnCrossGamma, _lnCrossBeta);
            }

            // Cross attention: Q = x, K/V = context
            var cross = _crossAttention.Forward(x, _context);

            // Residual + LayerNorm
            return ApplyResidualNorm(x, cross, _lnCrossGamma, _lnCrossBeta);
        }
    }
}