using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers
{
    public class TransformerGradients
    {
        public float[,] TokenEmbeddingGrad { get; set; }

        public float[,] InputProjectionGrad { get; set; }
        public float[] InputProjectionBiasGrad { get; set; }

        public List<AttentionGradients> AttentionGrads { get; set; }
        public List<LayerNormGradients> LN1Grads { get; set; }
        public List<LayerNormGradients> LN2Grads { get; set; }

        public float[,] OutputProjectionGrad { get; set; }
        public float[] OutputBiasGrad { get; set; }

        private readonly bool _usesDiscreteTokens;

        public TransformerGradients(TransformerConfig config)
        {
            _usesDiscreteTokens = config.UsesDiscreteTokens;

            if (_usesDiscreteTokens)
            {
                TokenEmbeddingGrad = new float[config.VocabSize, config.EmbeddingDim];
            }
            else
            {
                InputProjectionGrad = new float[config.EmbeddingDim, config.InputFeatureDim];
                InputProjectionBiasGrad = new float[config.EmbeddingDim];
            }

            AttentionGrads = new List<AttentionGradients>();
            LN1Grads = new List<LayerNormGradients>();
            LN2Grads = new List<LayerNormGradients>();

            for (int i = 0; i < config.NumLayers; i++)
            {
                AttentionGrads.Add(new AttentionGradients(config.EmbeddingDim));
                LN1Grads.Add(new LayerNormGradients(config.EmbeddingDim));
                LN2Grads.Add(new LayerNormGradients(config.EmbeddingDim));
            }

            int outputDim = config.EffectiveOutputDim;
            OutputProjectionGrad = new float[outputDim, config.EmbeddingDim];
            OutputBiasGrad = new float[outputDim];
        }

        public TransformerGradients(int numLayers, int embeddingDim, int vocabSize)
        {
            _usesDiscreteTokens = true;
            TokenEmbeddingGrad = new float[vocabSize, embeddingDim];

            AttentionGrads = new List<AttentionGradients>();
            LN1Grads = new List<LayerNormGradients>();
            LN2Grads = new List<LayerNormGradients>();

            for (int i = 0; i < numLayers; i++)
            {
                AttentionGrads.Add(new AttentionGradients(embeddingDim));
                LN1Grads.Add(new LayerNormGradients(embeddingDim));
                LN2Grads.Add(new LayerNormGradients(embeddingDim));
            }
            OutputProjectionGrad = new float[vocabSize, embeddingDim];
            OutputBiasGrad = new float[vocabSize];
        }

        public void Zero()
        {
            if (_usesDiscreteTokens)
            {
                ZeroMatrix(TokenEmbeddingGrad);
            }
            else
            {
                ZeroMatrix(InputProjectionGrad);
                Array.Clear(InputProjectionBiasGrad, 0, InputProjectionBiasGrad.Length);
            }

            foreach (var g in AttentionGrads)
            {
                g.Zero();
            }
            foreach (var g in LN1Grads)
            {
                g.Zero();
            }
            foreach (var g in LN2Grads)
            {
                g.Zero();
            }
            ZeroMatrix(OutputProjectionGrad);
            Array.Clear(OutputBiasGrad, 0, OutputBiasGrad.Length);
        }

        private void ZeroMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = 0;
                }
            }
        }
    }
}
