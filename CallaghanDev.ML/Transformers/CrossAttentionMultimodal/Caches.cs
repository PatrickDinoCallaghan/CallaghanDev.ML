using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using System.Collections.Generic;

namespace CallaghanDev.ML.Transformers.CrossAttentionMultimodal
{
    /// <summary>
    /// Caches all intermediate values from a cross-attention block forward pass,
    /// needed for backpropagation.
    /// </summary>
    public class CrossAttentionBlockCache
    {
        // Input to this block
        public float[,] BlockInput { get; set; }

        // === Self-Attention ===
        public float[,] SelfQ { get; set; }
        public float[,] SelfK { get; set; }
        public float[,] SelfV { get; set; }
        public float[,] SelfAttnOutput { get; set; }  // concatenated heads before WO
        public float[,] SelfResidualInput { get; set; } // x + selfProjected, input to LN
        public LayerNormCache LNSelfCache { get; set; }
        public float[,] NormedSelf { get; set; } // output of LN after self-attn

        // === Cross-Attention ===
        public float[,] CrossQ { get; set; }
        public float[,] CrossK { get; set; }
        public float[,] CrossV { get; set; }
        public float[,] CrossAttnOutput { get; set; }  // concatenated heads before WO
        public float[,] CrossResidualInput { get; set; } // normedSelf + crossProjected
        public LayerNormCache LNCrossCache { get; set; }
        public float[,] NormedCross { get; set; }

        // === FFN ===
        public float[][] FFNInputRows { get; set; }
        public float[,] FFNOutput { get; set; }
        public float[,] FFNResidualInput { get; set; } // normedCross + ffOutput
        public LayerNormCache LNFFNCache { get; set; }

        public CrossAttentionBlockCache()
        {
            LNSelfCache = new LayerNormCache();
            LNCrossCache = new LayerNormCache();
            LNFFNCache = new LayerNormCache();
        }
    }

    /// <summary>
    /// Gradients for a single CrossAttentionBlock's parameters.
    /// </summary>
    public class CrossAttentionBlockGradients
    {
        // Self-Attention gradients
        public AttentionGradients SelfAttnGrads { get; set; }
        public LayerNormGradients LNSelfGrads { get; set; }

        // Cross-Attention gradients
        public AttentionGradients CrossAttnGrads { get; set; }
        public LayerNormGradients LNCrossGrads { get; set; }

        // FFN gradients (managed externally via NeuralNetwork)
        public LayerNormGradients LNFFNGrads { get; set; }

        public CrossAttentionBlockGradients(int embeddingDim)
        {
            SelfAttnGrads = new AttentionGradients(embeddingDim);
            LNSelfGrads = new LayerNormGradients(embeddingDim);
            CrossAttnGrads = new AttentionGradients(embeddingDim);
            LNCrossGrads = new LayerNormGradients(embeddingDim);
            LNFFNGrads = new LayerNormGradients(embeddingDim);
        }

        public void Zero()
        {
            SelfAttnGrads.Zero();
            LNSelfGrads.Zero();
            CrossAttnGrads.Zero();
            LNCrossGrads.Zero();
            LNFFNGrads.Zero();
        }
    }

    /// <summary>
    /// Forward cache for the entire multimodal model.
    /// </summary>
    public class MultimodalForwardCache
    {
        // === Text Encoder ===
        public float[,] TextEmbedded { get; set; }
        public int[] TextTokenIds { get; set; }
        public List<float[,]> TextLayerInputs { get; set; }
        public List<AttentionCache> TextAttentionCaches { get; set; }
        public List<LayerNormCache> TextLN1Caches { get; set; }
        public List<LayerNormCache> TextLN2Caches { get; set; }
        public List<float[,]> TextFFNOutputs { get; set; }
        public List<float[][]> TextFFNInputs { get; set; }
        public float[,] TextFinalHidden { get; set; }

        // === Price Decoder (with cross-attention) ===
        public float[,] PriceEmbedded { get; set; }
        public float[,] PriceContinuousInput { get; set; }
        public List<CrossAttentionBlockCache> PriceBlockCaches { get; set; }
        public float[,] PriceFinalHidden { get; set; }

        public MultimodalForwardCache(int textNumLayers, int priceNumLayers)
        {
            TextLayerInputs = new List<float[,]>();
            TextAttentionCaches = new List<AttentionCache>();
            TextLN1Caches = new List<LayerNormCache>();
            TextLN2Caches = new List<LayerNormCache>();
            TextFFNOutputs = new List<float[,]>();
            TextFFNInputs = new List<float[][]>();

            for (int i = 0; i < textNumLayers; i++)
            {
                TextAttentionCaches.Add(new AttentionCache());
                TextLN1Caches.Add(new LayerNormCache());
                TextLN2Caches.Add(new LayerNormCache());
            }

            PriceBlockCaches = new List<CrossAttentionBlockCache>();
            for (int i = 0; i < priceNumLayers; i++)
            {
                PriceBlockCaches.Add(new CrossAttentionBlockCache());
            }
        }
    }

}
