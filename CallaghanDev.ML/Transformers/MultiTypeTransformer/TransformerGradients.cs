using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MultiTypeTransformer
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
        private readonly SparseRowGradientTracker _tokenEmbeddingRows = new SparseRowGradientTracker();

        internal SparseRowGradientTracker TokenEmbeddingRows => _tokenEmbeddingRows;

        public TransformerGradients(TransformerConfig config)
        {
            _usesDiscreteTokens = config.Data.UsesDiscreteTokens;

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
                _tokenEmbeddingRows.ZeroTrackedRowsAndClear(TokenEmbeddingGrad);
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


        public void MarkTokenRows(int[] tokenIds, int tokenStart, int tokenCount)
        {
            if (!_usesDiscreteTokens)
            {
                return;
            }

            _tokenEmbeddingRows.MarkRows(tokenIds, tokenStart, tokenCount, TokenEmbeddingGrad.GetLength(0));
        }

        public float TokenEmbeddingSquaredNorm()
        {
            return _usesDiscreteTokens ? _tokenEmbeddingRows.SquaredNorm(TokenEmbeddingGrad) : 0f;
        }

        public void ScaleTokenEmbeddingGrad(float scale)
        {
            if (_usesDiscreteTokens)
            {
                _tokenEmbeddingRows.Scale(TokenEmbeddingGrad, scale);
            }
        }

        public void UpdateTokenEmbedding(float[,] tokenEmbedding, float learningRate)
        {
            if (_usesDiscreteTokens)
            {
                _tokenEmbeddingRows.UpdateRows(tokenEmbedding, TokenEmbeddingGrad, learningRate);
            }
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
