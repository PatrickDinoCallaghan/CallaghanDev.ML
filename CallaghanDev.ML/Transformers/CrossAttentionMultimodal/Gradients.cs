using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.CrossAttentionMultimodal
{
    public class Gradients
    {
        public float[,] TextEmbeddingGrad { get; set; }
        public List<AttentionGradients> TextAttnGrads { get; set; }
        public List<LayerNormGradients> TextLN1Grads { get; set; }
        public List<LayerNormGradients> TextLN2Grads { get; set; }

        public float[,] PriceInputProjectionGrad { get; set; }
        public float[] PriceInputProjectionBiasGrad { get; set; }
        public List<CrossAttentionBlockGradients> PriceBlockGrads { get; set; }

        public float[,] OutputProjectionGrad { get; set; }
        public float[] OutputBiasGrad { get; set; }

        public float[,] ConfidenceProjectionGrad { get; set; }
        public float[] ConfidenceBiasGrad { get; set; }

        public Gradients(Config config)
        {
            // Text encoder
            TextEmbeddingGrad = new float[config.TextVocabSize, config.TextEmbeddingDim];
            TextAttnGrads = new List<AttentionGradients>();
            TextLN1Grads = new List<LayerNormGradients>();
            TextLN2Grads = new List<LayerNormGradients>();
            for (int i = 0; i < config.TextNumLayers; i++)
            {
                TextAttnGrads.Add(new AttentionGradients(config.TextEmbeddingDim));
                TextLN1Grads.Add(new LayerNormGradients(config.TextEmbeddingDim));
                TextLN2Grads.Add(new LayerNormGradients(config.TextEmbeddingDim));
            }

            // Price decoder
            PriceInputProjectionGrad = new float[config.PriceEmbeddingDim, config.PriceInputFeatureDim];
            PriceInputProjectionBiasGrad = new float[config.PriceEmbeddingDim];
            PriceBlockGrads = new List<CrossAttentionBlockGradients>();
            for (int i = 0; i < config.PriceNumLayers; i++)
            {
                PriceBlockGrads.Add(new CrossAttentionBlockGradients(config.PriceEmbeddingDim));
            }

            // Output head
            OutputProjectionGrad = new float[config.OutputDim, config.PriceEmbeddingDim];
            OutputBiasGrad = new float[config.OutputDim];

            if (config.UseConfidenceHead)
            {
                ConfidenceProjectionGrad = new float[1, config.PriceEmbeddingDim];
                ConfidenceBiasGrad = new float[1];
            }
        }

        public void Zero()
        {
            ZeroMatrix(TextEmbeddingGrad);
            foreach (var g in TextAttnGrads) g.Zero();
            foreach (var g in TextLN1Grads) g.Zero();
            foreach (var g in TextLN2Grads) g.Zero();

            ZeroMatrix(PriceInputProjectionGrad);
            System.Array.Clear(PriceInputProjectionBiasGrad, 0, PriceInputProjectionBiasGrad.Length);
            foreach (var g in PriceBlockGrads) g.Zero();

            ZeroMatrix(OutputProjectionGrad);
            System.Array.Clear(OutputBiasGrad, 0, OutputBiasGrad.Length);

            if (ConfidenceProjectionGrad != null)
            {
                ZeroMatrix(ConfidenceProjectionGrad);
                System.Array.Clear(ConfidenceBiasGrad, 0, ConfidenceBiasGrad.Length);
            }
        }

        private void ZeroMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = 0;
        }
    }
}
