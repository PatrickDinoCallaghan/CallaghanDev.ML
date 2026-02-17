using System;
using System.Collections.Generic;

namespace CallaghanDev.ML.Transformers.TACAMT
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

        /// <summary>
        /// Gradient accumulator for context type embeddings.
        /// Shape: [2, PriceEmbeddingDim]
        /// Index 0 = news context gradient, Index 1 = price memory context gradient.
        /// </summary>
        public float[,] ContextTypeEmbeddingGrad { get; set; }

        public Gradients(Config config)
        {
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

            PriceInputProjectionGrad = new float[config.PriceEmbeddingDim, config.PriceInputFeatureDim];
            PriceInputProjectionBiasGrad = new float[config.PriceEmbeddingDim];
            PriceBlockGrads = new List<CrossAttentionBlockGradients>();

            for (int i = 0; i < config.PriceNumLayers; i++)
            {
                PriceBlockGrads.Add(new CrossAttentionBlockGradients(config.PriceEmbeddingDim, config.PriceNumHeads, config.DecayProjectionDim, config.DecayHiddenDim, config.DecayTimeEncodingBases));
            }

            OutputProjectionGrad = new float[config.OutputDim, config.PriceEmbeddingDim];
            OutputBiasGrad = new float[config.OutputDim];

            if (config.UseConfidenceHead)
            { 
                ConfidenceProjectionGrad = new float[1, config.PriceEmbeddingDim]; 
                ConfidenceBiasGrad = new float[1]; 
            }

            ContextTypeEmbeddingGrad = new float[2, config.PriceEmbeddingDim];
        }

        public void Zero()
        {
            ZM(TextEmbeddingGrad);
            foreach (var g in TextAttnGrads)
            {
                g.Zero();
            }
            foreach (var g in TextLN1Grads)
            {
                g.Zero(); 
            }
            foreach (var g in TextLN2Grads)
            {
                g.Zero();
            }
            
            ZM(PriceInputProjectionGrad);
            Array.Clear(PriceInputProjectionBiasGrad, 0, PriceInputProjectionBiasGrad.Length);
            foreach (var g in PriceBlockGrads)
            {
                g.Zero();
            }
            ZM(OutputProjectionGrad);

            Array.Clear(OutputBiasGrad, 0, OutputBiasGrad.Length);

            if (ConfidenceProjectionGrad != null) 
            { 
                ZM(ConfidenceProjectionGrad); 
                Array.Clear(ConfidenceBiasGrad, 0, ConfidenceBiasGrad.Length);
            }

            ZM(ContextTypeEmbeddingGrad);
        }

        private void ZM(float[,] m) 
        { 
            int r = m.GetLength(0), c = m.GetLength(1);

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    m[i, j] = 0;
                }
            }
        }
    }
}