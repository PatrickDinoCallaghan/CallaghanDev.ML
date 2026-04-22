using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.TACAMT;

namespace CallaghanDev.ML.Transformers.MMTAC
{
    public class MmtacForwardCache
    {
        // Text encoder
        public float[,] TextEmbedded { get; set; }
        public int[] TextTokenIds { get; set; }
        public List<float[,]> TextLayerInputs { get; set; }
        public List<AttentionCache> TextAttentionCaches { get; set; }
        public List<LayerNormCache> TextLN1Caches { get; set; }
        public List<LayerNormCache> TextLN2Caches { get; set; }
        public List<float[,]> TextFFNOutputs { get; set; }
        public List<float[,]> TextFFNInputs { get; set; }
        public float[,] TextFinalHidden { get; set; }

        // Multi-story
        public List<MmtacForwardCache> StoryCaches { get; set; }
        public int[] StoryTokenCounts { get; set; }
        public float[] StoryArrivalTimes { get; set; }

        // Global token
        public float[] GlobalTokenEmbedded { get; set; }
        /// <summary>Raw (pre-projection) global feature vector. Stored for backward.</summary>
        public float[] GlobalRawInput { get; set; }

        // Price decoder
        public float[,] PriceEmbedded { get; set; }
        public float[,] PriceContinuousInput { get; set; }
        public List<BlockCache> PriceBlockCaches { get; set; }
        public float[,] PriceFinalHidden { get; set; }

        // Context metadata
        public int NumNewsContext { get; set; } = 0;
        public int NumPriceContext { get; set; } = 0;
        public int NumGlobalContext { get; set; } = 0;
        public float[,] PriceContextHidden { get; set; }

        // Pre-activation logits for non-linear output heads (needed for backward)
        /// <summary>Pre-softplus logits for Range head [seqLen].  Set by ProjectToOutputs.</summary>
        public float[] RangeLogits { get; set; }
        /// <summary>Pre-sigmoid logits for Quality head [seqLen].  Set by ProjectToOutputs.</summary>
        public float[] QualityLogits { get; set; }
        public int NumStoredNewsContext { get; set; } = 0;
        public int NumLiveNewsContext { get; set; } = 0;
        public MmtacForwardCache(int textNumLayers, int priceNumLayers)
        {
            TextLayerInputs = new List<float[,]>();
            TextAttentionCaches = new List<AttentionCache>();
            TextLN1Caches = new List<LayerNormCache>();
            TextLN2Caches = new List<LayerNormCache>();
            TextFFNOutputs = new List<float[,]>();
            TextFFNInputs = new List<float[,]>();

            for (int i = 0; i < textNumLayers; i++)
            {
                TextAttentionCaches.Add(new AttentionCache());
                TextLN1Caches.Add(new LayerNormCache());
                TextLN2Caches.Add(new LayerNormCache());
            }

            PriceBlockCaches = new List<BlockCache>();

            for (int i = 0; i < priceNumLayers; i++)
            {
                PriceBlockCaches.Add(new BlockCache());
            }
        }

        public void Reset()
        {
            TextEmbedded = null;
            TextTokenIds = null;
            TextFinalHidden = null;
            GlobalTokenEmbedded = null;
            GlobalRawInput = null;
            RangeLogits = null;
            QualityLogits = null;

            TextLayerInputs.Clear();
            TextFFNOutputs.Clear();
            TextFFNInputs.Clear();

            for (int i = 0; i < TextAttentionCaches.Count; i++)
            {
                TextAttentionCaches[i].Reset();
                TextLN1Caches[i].Reset();
                TextLN2Caches[i].Reset();
            }

            if (StoryCaches != null)
            {
                foreach (var sc in StoryCaches)
                {
                    sc?.Reset();
                }
                StoryCaches.Clear();
            }

            StoryTokenCounts = null;
            StoryArrivalTimes = null;
            PriceEmbedded = null;
            PriceContinuousInput = null;
            PriceFinalHidden = null;
            PriceContextHidden = null;

            NumNewsContext = 0;
            NumPriceContext = 0;
            NumGlobalContext = 0;
            NumStoredNewsContext = 0;
            NumLiveNewsContext = 0;

            for (int i = 0; i < PriceBlockCaches.Count; i++)
            {
                PriceBlockCaches[i].Reset();
            }
        }
    }
}