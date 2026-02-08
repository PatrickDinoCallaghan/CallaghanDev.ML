using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers
{
    public class TransformerBlock
    {
        public MultiHeadAttention Attention { get; set; }
        public NeuralNetwork FeedForwardNetwork { get; set; }

        public float[] LN1Gamma { get; set; }
        public float[] LN1Beta { get; set; }
        public float[] LN2Gamma { get; set; }
        public float[] LN2Beta { get; set; }

        private readonly int _embeddingDim;
        private readonly IAccelerationManager _accel;

        public TransformerBlock(TransformerConfig config, IAccelerationManager accel, Random random = null)
        {
            _embeddingDim = config.EmbeddingDim;
            _accel = accel;

            Attention = new MultiHeadAttention(config.EmbeddingDim, config.NumHeads, accel, random);

            var ffnParams = new Parameters
            {
                LayerWidths = new List<int>
                {
                    config.EmbeddingDim,
                    config.FeedForwardDim,
                    config.EmbeddingDim
                },
                LayerActivations = new List<ActivationType>
                {
                    ActivationType.None,
                    config.FFNActivationType,
                    ActivationType.None
                },
                AccelerationType = config.AccelerationType,
                AccelerationDeviceId = config.AccelerationDeviceId,
                CostFunction = config.CostFunction,
                ActivationDistribution = config.ActivationDistribution,
                L2RegulationLamda = config.L2RegulationLamda,
                GradientClippingThreshold = config.GradientClippingThreshold,
                inputActivationMin = Enumerable.Repeat(0f, config.EmbeddingDim).ToArray(),
                inputActivationMax = Enumerable.Repeat(1f, config.EmbeddingDim).ToArray()
            };

            FeedForwardNetwork = new NeuralNetwork(ffnParams);

            LN1Gamma = Enumerable.Repeat(1.0f, config.EmbeddingDim).ToArray();
            LN1Beta = new float[config.EmbeddingDim];
            LN2Gamma = Enumerable.Repeat(1.0f, config.EmbeddingDim).ToArray();
            LN2Beta = new float[config.EmbeddingDim];
        }

        public float[,] Forward(float[,] input, bool[,] mask = null)
        {
            int seqLen = input.GetLength(0);

            var attnOutput = Attention.Forward(input, mask);
            var attnResidual = _accel.MatrixAdd(input, attnOutput);
            var normed1 = _accel.LayerNorm(attnResidual, LN1Gamma, LN1Beta);

            var ffOutput = new float[seqLen, _embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                var inputRow = new float[_embeddingDim];
                for (int j = 0; j < _embeddingDim; j++)
                {
                    inputRow[j] = normed1[i, j];
                }

                var outputRow = FeedForwardNetwork.Predict(inputRow);

                for (int j = 0; j < _embeddingDim; j++)
                {
                    ffOutput[i, j] = outputRow[j];
                }
            }

            var ffResidual = _accel.MatrixAdd(normed1, ffOutput);
            return _accel.LayerNorm(ffResidual, LN2Gamma, LN2Beta);
        }
    }
}
