using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using System;

namespace CallaghanDev.ML.Transformers.TACAMT
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

        /// <summary>
        /// Content-aware decay network v3.
        /// Learns decay = f(query_content, key_content, time, other_memories).
        /// Now includes dropout on memory attention and MLP hidden, plus L2 weight decay.
        /// </summary>
        public ContentAwareDecayNetwork DecayNetwork { get; set; }

        private readonly int _embeddingDim;
        private readonly int _numHeads;
        private readonly int _headDim;

        private readonly IAccelerationManager _accel;

        public TransformerBlock(int embeddingDim, int numHeads, int feedForwardDim, ActivationType ffnActivation, IAccelerationManager accel, Random random, AccelerationType accelType = AccelerationType.CPU, int accelDeviceId = 0, float l2Lambda = 0.01f, int decayProjectionDim = 8, int decayHiddenDim = 16, float decayMemAttnDropout = 0.1f, float decayMLPDropout = 0.1f, float decayWeightDecay = 0.0f, int decayTimeBases = 8)
        {
            _accel = accel;
            _numHeads = numHeads;
            _embeddingDim = embeddingDim;
            _headDim = embeddingDim / numHeads;

            SelfAttention = new MultiHeadAttention(embeddingDim, _numHeads, accel, random);
            LNSelfGamma = new float[embeddingDim];
            LNSelfBeta = new float[embeddingDim];

            CrossAttention = new MultiHeadAttention(embeddingDim, _numHeads, accel, random);
            LNCrossGamma = new float[embeddingDim];
            LNCrossBeta = new float[embeddingDim];

            DecayNetwork = new ContentAwareDecayNetwork(numHeads, embeddingDim, decayProjectionDim, decayHiddenDim, random, decayMemAttnDropout, decayMLPDropout, decayWeightDecay, decayTimeBases);

            var parameters = new Parameters
            {
                AccelerationType = accelType,
                CostFunction = CostFunctionType.mse,
                ActivationDistribution = ActivationDistribution.Normal,
                LayerWidths = new List<int> { embeddingDim, feedForwardDim, embeddingDim },
                LayerActivations = new List<ActivationType> { ffnActivation, ffnActivation, ffnActivation },
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

        /// <summary>
        /// Forward pass with content-aware time-decay cross-attention.
        /// </summary>
        /// <param name="x">Price hidden states [priceSeqLen, embDim]</param>
        /// <param name="contextHidden">Context hidden states [numKeys, embDim], or null</param>
        /// <param name="selfAttnMask">Causal mask for self-attention</param>
        /// <param name="accel">Acceleration manager</param>
        /// <param name="timeDiffs">[priceSeqLen, numKeys] absolute time differences, or null</param>
        /// <param name="keyTimesFromRef">[numKeys] each key's time relative to reference, for memory interaction</param>
        /// <param name="isTraining">Whether to apply dropout (true during training, false at inference)</param>
        /// <param name="dropoutRng">Random number generator for dropout masks</param>
        public float[,] Forward(float[,] x, float[,] contextHidden, bool[,] selfAttnMask, IAccelerationManager accel, float[,] timeDiffs = null, float[] keyTimesFromRef = null, bool isTraining = false, Random dropoutRng = null)
        {
            int priceSeqLen = x.GetLength(0);
            int numHeads = _numHeads;
            int headDim = _headDim;

            float scale = 1.0f / MathF.Sqrt(headDim);

            var selfQ = ComputeProjection(x, SelfAttention.WQ, SelfAttention.BiasQ, accel);
            var selfK = ComputeProjection(x, SelfAttention.WK, SelfAttention.BiasK, accel);
            var selfV = ComputeProjection(x, SelfAttention.WV, SelfAttention.BiasV, accel);

            var selfAttnOut = accel.MultiHeadAttentionForward(selfQ, selfK, selfV, numHeads, scale, selfAttnMask);

            var selfProjected = ComputeProjection(selfAttnOut, SelfAttention.WO, SelfAttention.BiasO, accel);
            var selfResidual = accel.MatrixAdd(x, selfProjected);

            var (normedSelf, _, _, _) = accel.LayerNormForward(
                selfResidual,
                LNSelfGamma,
                LNSelfBeta
            );

            float[,] normedCross;

            if (contextHidden != null)
            {
                float[,] crossAttnOut;

                if (timeDiffs != null)
                {
                    crossAttnOut = ContentAwareCrossAttention(normedSelf, contextHidden, timeDiffs, keyTimesFromRef, accel, isTraining, dropoutRng);
                }
                else
                {
                    var crossQ = ComputeProjection(normedSelf, CrossAttention.WQ, CrossAttention.BiasQ, accel);
                    var crossK = ComputeProjection(contextHidden, CrossAttention.WK, CrossAttention.BiasK, accel);
                    var crossV = ComputeProjection(contextHidden, CrossAttention.WV, CrossAttention.BiasV, accel);

                    crossAttnOut = accel.MultiHeadAttentionForward(
                        crossQ,
                        crossK,
                        crossV,
                        numHeads,
                        scale,
                        null
                    );
                }

                var crossProjected = ComputeProjection(crossAttnOut, CrossAttention.WO, CrossAttention.BiasO, accel);
                var crossResidual = accel.MatrixAdd(normedSelf, crossProjected);

                var (nc, _, _, _) = accel.LayerNormForward(
                    crossResidual,
                    LNCrossGamma,
                    LNCrossBeta
                );

                normedCross = nc;
            }
            else
            {
                var (nc, _, _, _) = accel.LayerNormForward(normedSelf, LNCrossGamma, LNCrossBeta);

                normedCross = nc;
            }

            /*var ffOutput = new float[priceSeqLen, _embeddingDim];

            for (int i = 0; i < priceSeqLen; i++)
            {
                var inputRow = new float[_embeddingDim];

                for (int j = 0; j < _embeddingDim; j++)
                {
                    inputRow[j] = normedCross[i, j];
                }

                var outputRow = FeedForwardNetwork.ForwardPassOnly(inputRow);

                for (int j = 0; j < _embeddingDim; j++)
                {
                    ffOutput[i, j] = outputRow[j];
                }ComputeTimeDiffMatrix
            }*/

            var ffOutput = accel.FFNForwardBatch(normedCross, priceSeqLen, _embeddingDim, FeedForwardNetwork.ForwardPassOnly);

            var ffResidual = accel.MatrixAdd(normedCross, ffOutput);

            var (normedFF, _, _, _) = accel.LayerNormForward(ffResidual, LNFFNGamma, LNFFNBeta);

            return normedFF;
        }


        public float[,] ComputeTimeDiffMatrix(int priceSeqLen, float[] keyArrivalTimes)
        {
            return _accel.ComputeTimeDiffMatrix(priceSeqLen, keyArrivalTimes);
            /*
            int numKeys = keyArrivalTimes.Length;
            var timeDiffs = new float[priceSeqLen, numKeys];

            for (int p = 0; p < priceSeqLen; p++)
            {
                for (int s = 0; s < numKeys; s++)
                {
                    timeDiffs[p, s] = MathF.Abs(p - keyArrivalTimes[s]);
                }
            }

            return timeDiffs;*/
        }

        private float[,] ContentAwareCrossAttention(float[,] priceHidden, float[,] contextHidden, float[,] timeDiffs, float[] keyTimesFromRef, IAccelerationManager accel, bool isTraining = false, Random dropoutRng = null)
        {
            int priceSeqLen = priceHidden.GetLength(0);
            int contextSeqLen = contextHidden.GetLength(0);
            int embDim = _embeddingDim;
            int numHeads = _numHeads;
            int headDim = _headDim;

            float scale = 1.0f / MathF.Sqrt(headDim);

            var Q = ComputeProjection(priceHidden, CrossAttention.WQ, CrossAttention.BiasQ, accel);
            var K = ComputeProjection(contextHidden, CrossAttention.WK, CrossAttention.BiasK, accel);
            var V = ComputeProjection(contextHidden, CrossAttention.WV, CrossAttention.BiasV, accel);

            //var (decayBias, _) = DecayNetwork.Forward(priceHidden, contextHidden, timeDiffs, keyTimesFromRef, isTraining, dropoutRng);

            var (decayBias, _) = accel.ContentAwareDecayForward(priceHidden, contextHidden, timeDiffs, keyTimesFromRef, DecayNetwork, isTraining, dropoutRng);

            /* var output = new float[priceSeqLen, embDim];
           
            for (int h = 0; h < numHeads; h++)
            {
                int startIdx = h * headDim;

                for (int p = 0; p < priceSeqLen; p++)
                {
                    var scores = new float[contextSeqLen];
                    float maxScore = float.MinValue;

                    for (int s = 0; s < contextSeqLen; s++)
                    {
                        float dot = 0;

                        for (int d = 0; d < headDim; d++)
                        {
                            dot += Q[p, startIdx + d] * K[s, startIdx + d];
                        }

                        scores[s] = dot * scale + decayBias[p, s, h];

                        if (scores[s] > maxScore)
                        {
                            maxScore = scores[s];
                        }
                    }

                    float sumExp = 0;
                    var w = new float[contextSeqLen];

                    for (int s = 0; s < contextSeqLen; s++)
                    {
                        w[s] = MathF.Exp(scores[s] - maxScore);
                        sumExp += w[s];
                    }

                    if (sumExp > 0)
                    {
                        for (int s = 0; s < contextSeqLen; s++)
                        {
                            w[s] /= sumExp;
                        }
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        float val = 0;

                        for (int s = 0; s < contextSeqLen; s++)
                        {
                            val += w[s] * V[s, startIdx + d];
                        }

                        output[p, startIdx + d] = val;
                    }
                }
            }
            */

            return accel.ContentAwareCrossAttentionForward(Q, K, V, numHeads, scale, decayBias, out _, out _);

            //return output;
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
