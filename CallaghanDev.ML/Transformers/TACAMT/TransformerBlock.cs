using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using System;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public class TacamtBlock : TransformerBlockBase
    {
        public readonly MultiHeadAttention CrossAttention;

        public readonly RotaryPositionEmbedding _rotaryPositionEmbedding;

        public float[] LnCrossGamma;
        public float[] LnCrossBeta;
        public ContentAwareDecayNetwork DecayNetwork;

        private float[,] _context;
        private float[,] _timeDiffs;
        private float[] _keyTimes;

        private bool _isTraining;
        private bool _useDecayNetwork = true;
        private Random _dropoutRng;
        public bool UseDecayNetworkForCrossAttention => _useDecayNetwork;
        public TacamtBlock(int embeddingDim, int numHeads, int feedForwardDim, ActivationType ffnActivation, IAccelerationManager accel, Random random,  float l2Lambda = 0.01f, int decayProjectionDim = 8, int decayHiddenDim = 16, float decayMemAttnDropout = 0.1f, float decayMLPDropout = 0.1f, float decayWeightDecay = 0.0f, int decayTimeBases = 8) : base(embeddingDim, numHeads, accel)
        {
            _rotaryPositionEmbedding = new RotaryPositionEmbedding(accel);
            SelfAttention = new MultiHeadAttention(embeddingDim, numHeads, accel, random);

            LNSelfGamma = InitGamma(embeddingDim);
            LNSelfBeta = new float[embeddingDim];

            CrossAttention = new MultiHeadAttention(embeddingDim, numHeads, accel, random);

            LnCrossGamma = InitGamma(embeddingDim);
            LnCrossBeta = new float[embeddingDim];

            DecayNetwork = new ContentAwareDecayNetwork(
                numHeads,
                embeddingDim,
                decayProjectionDim,
                decayHiddenDim,
                random,
                decayMemAttnDropout,
                decayMLPDropout,
                decayWeightDecay,
                decayTimeBases);

            FeedForwardNetwork = new NeuralNetwork(new Parameters
            {
                LayerWidths = new List<int> { embeddingDim, feedForwardDim, embeddingDim },
                LayerActivations = new List<ActivationType> { ffnActivation, ffnActivation, ActivationType.None },
                L2RegulationLamda = l2Lambda
            });

            LNFFNGamma = InitGamma(embeddingDim);
            LNFFNBeta = new float[embeddingDim];
        }


        // ---- External setters (cleaner than giant Forward signature) ----
        public void SetContext(float[,] context) => _context = context;
        public void SetTimeData(float[,] timeDiffs, float[] keyTimes)
        {
            _timeDiffs = timeDiffs;
            _keyTimes = keyTimes;
        }

        public void SetTraining(bool isTraining, Random rng = null)
        {
            _isTraining = isTraining;
            _dropoutRng = rng;
        }

        // ---- Cross Attention Hook ----
        protected override float[,] CrossAttentionForward(float[,] x)
        {
            if (_context == null)
            {
                return Accel.LayerNorm(x, LnCrossGamma, LnCrossBeta);
            }

            float[,] cross;

            if (_timeDiffs != null)
            {
                cross = ContentAwareCrossAttention(x);
            }
            else
            {
                cross = CrossAttention.Forward(x, _context);
            }

            return ApplyResidualNorm(x, cross, LnCrossGamma, LnCrossBeta);
        }
        private float[,] ContentAwareCrossAttention(float[,] x)
        {
            var Q = ComputeProjection(x, CrossAttention.WQ, CrossAttention.BiasQ, Accel);
            var K = ComputeProjection(_context, CrossAttention.WK, CrossAttention.BiasK, Accel);
            var V = ComputeProjection(_context, CrossAttention.WV, CrossAttention.BiasV, Accel);

            _rotaryPositionEmbedding.ApplyInPlace(Q, K, NumHeads);

            float scale = 1.0f / MathF.Sqrt(HeadDim);
            float[,,] attentionBias = null;
            bool[] hasValidKey = ComputeHasValidCrossAttentionKey(_timeDiffs, x.GetLength(0), _context.GetLength(0));

            if (_timeDiffs != null && _useDecayNetwork && DecayNetwork != null)
            {
                var (decayBias, _) = Accel.ContentAwareDecayForward(
                    x,
                    _context,
                    _timeDiffs,
                    _keyTimes,
                    DecayNetwork,
                    _isTraining,
                    _dropoutRng);

                attentionBias = decayBias;
            }
            else if (_timeDiffs != null)
            {
                int queryLen = _timeDiffs.GetLength(0);
                int keyLen = _timeDiffs.GetLength(1);

                attentionBias = new float[queryLen, keyLen, NumHeads];

                for (int q = 0; q < queryLen; q++)
                {
                    for (int s = 0; s < keyLen; s++)
                    {
                        float value = _timeDiffs[q, s] < 0f
                            ? float.NegativeInfinity
                            : 0f;

                        for (int h = 0; h < NumHeads; h++)
                            attentionBias[q, s, h] = value;
                    }
                }
            }

            var attnOutput = Accel.ContentAwareCrossAttentionForward(
                Q,
                K,
                V,
                NumHeads,
                scale,
                attentionBias,
                out _,
                out _);

            return ComputeProjectionWithOptionalRows(
                attnOutput,
                CrossAttention.WO,
                CrossAttention.BiasO,
                hasValidKey,
                Accel);
        }
        private float[] InitGamma(int size)
        {
            var g = new float[size];
            for (int i = 0; i < size; i++) g[i] = 1f;
            return g;
        }
        public void SetTimeData(float[,] timeDiffs, float[] keyTimes, bool useDecayNetwork = true)
        {
            _timeDiffs = timeDiffs;
            _keyTimes = keyTimes;
            _useDecayNetwork = useDecayNetwork;
        }
        private float[,] ComputeProjectionWithOptionalRows(
    float[,] input,
    float[,] weight,
    float[] bias,
    bool[] includeRows,
    IAccelerationManager accel)
        {
            var projected = accel.BatchDotProduct(weight, input);
            int rows = projected.GetLength(0);
            int cols = projected.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                if (includeRows != null && !includeRows[i])
                    continue;

                for (int j = 0; j < cols; j++)
                    result[i, j] = projected[i, j] + bias[j];
            }

            return result;
        }
        private static bool[] ComputeHasValidCrossAttentionKey(float[,] timeDiffs, int queryLen, int keyLen)
        {
            var hasValidKey = new bool[queryLen];

            if (timeDiffs == null)
            {
                for (int q = 0; q < queryLen; q++)
                    hasValidKey[q] = keyLen > 0;

                return hasValidKey;
            }

            for (int q = 0; q < queryLen; q++)
            {
                for (int s = 0; s < keyLen; s++)
                {
                    if (timeDiffs[q, s] >= 0f)
                    {
                        hasValidKey[q] = true;
                        break;
                    }
                }
            }

            return hasValidKey;
        }

        private float[,] ComputeProjection(float[,] input, float[,] weight, float[] bias, IAccelerationManager accel)
        {
            var projected = accel.BatchDotProduct(weight, input);
            int rows = projected.GetLength(0), cols = projected.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = projected[i, j] + bias[j];
                }
            }

            return result;
        }
    }
}
