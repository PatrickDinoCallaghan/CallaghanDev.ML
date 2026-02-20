using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
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
        public TransformerBlock(int embeddingDim, int numHeads, int feedForwardDim, ActivationType ffnActivation, IAccelerationManager accel, Random random, AccelerationType accelType = AccelerationType.CPU, int accelDeviceId = 0, float l2Lambda = 0.01f)
        {
            if (embeddingDim % numHeads != 0)
            {
                throw new InvalidOperationException($"EmbeddingDim ({embeddingDim}) must be divisible by NumHeads ({numHeads})");
            }
            _numHeads = numHeads;
            _embeddingDim = embeddingDim;

            //Self attention
            SelfAttention = new MultiHeadAttention(embeddingDim, _numHeads, accel, random);
            LNSelfGamma = new float[embeddingDim];
            LNSelfBeta = new float[embeddingDim];

            //Cross attention
            // Uses the SAME MultiHeadAttention structure, but during forward:
            //   Q comes from price hidden states
            //   K, V come from text encoder output
            CrossAttention = new MultiHeadAttention(embeddingDim, _numHeads, accel, random);
            LNCrossGamma = new float[embeddingDim];
            LNCrossBeta = new float[embeddingDim];


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

            // Initialize all LayerNorm gammas to 1
            for (int i = 0; i < embeddingDim; i++)
            {
                LNSelfGamma[i] = 1.0f;
                LNCrossGamma[i] = 1.0f;
                LNFFNGamma[i] = 1.0f;
            }
        }

        public float[,] Forward(float[,] x, float[,] textHidden, bool[,] selfAttnMask, IAccelerationManager accel)
        {
            int priceSeqLen = x.GetLength(0);
            int numHeads = _numHeads;
            int headDim = _embeddingDim / numHeads;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Self-Attention (price => price)
            var selfQ = ComputeProjection(x, SelfAttention.WQ, SelfAttention.BiasQ, accel);
            var selfK = ComputeProjection(x, SelfAttention.WK, SelfAttention.BiasK, accel);
            var selfV = ComputeProjection(x, SelfAttention.WV, SelfAttention.BiasV, accel);

            var selfAttnOut = accel.MultiHeadAttentionForward(selfQ, selfK, selfV, numHeads, scale, selfAttnMask);
            var selfProjected = ComputeProjection(selfAttnOut, SelfAttention.WO, SelfAttention.BiasO, accel);

            // Residual + LayerNorm
            var selfResidual = accel.MatrixAdd(x, selfProjected);
            var (normedSelf, _, _, _) = accel.LayerNormForward(selfResidual, LNSelfGamma, LNSelfBeta);

            // Cross-Attention: SKIP if no text available
            float[,] normedCross;
            if (textHidden != null)
            {
                var crossQ = ComputeProjection(normedSelf, CrossAttention.WQ, CrossAttention.BiasQ, accel);
                var crossK = ComputeProjection(textHidden, CrossAttention.WK, CrossAttention.BiasK, accel);
                var crossV = ComputeProjection(textHidden, CrossAttention.WV, CrossAttention.BiasV, accel);

                var crossAttnOut = accel.MultiHeadAttentionForward(crossQ, crossK, crossV, numHeads, scale, null);
                var crossProjected = ComputeProjection(crossAttnOut, CrossAttention.WO, CrossAttention.BiasO, accel);

                var crossResidual = accel.MatrixAdd(normedSelf, crossProjected);
                var (normedCrossResult, _, _, _) = accel.LayerNormForward(crossResidual, LNCrossGamma, LNCrossBeta);
                normedCross = normedCrossResult;
            }
            else
            {
                // No text: pass through cross-attention LN as identity-like
                // (still apply LN so the FFN sees consistent scale)
                var (normedCrossResult, _, _, _) = accel.LayerNormForward(normedSelf, LNCrossGamma, LNCrossBeta);
                normedCross = normedCrossResult;
            }

            // FFN
           /* var ffOutput = new float[priceSeqLen, _embeddingDim];
            for (int i = 0; i < priceSeqLen; i++)
            {
                var inputRow = new float[_embeddingDim];
                for (int j = 0; j < _embeddingDim; j++)
                    inputRow[j] = normedCross[i, j];

                var outputRow = FeedForwardNetwork.ForwardPassOnly(inputRow);
                for (int j = 0; j < _embeddingDim; j++)
                    ffOutput[i, j] = outputRow[j];
            }*/

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
}