using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using System.Collections.Generic;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public class BlockCache
    {
        public float[,] BlockInput { get; set; }

        // Self Attention
        public float[,] SelfQ { get; set; }
        public float[,] SelfK { get; set; }
        public float[,] SelfV { get; set; }
        public float[,] SelfAttnOutput { get; set; }
        public float[,] SelfResidualInput { get; set; }
        public LayerNormCache LNSelfCache { get; set; }
        public float[,] NormedSelf { get; set; }

        // Cross Attention
        public float[,] CrossQ { get; set; }
        public float[,] CrossK { get; set; }
        public float[,] CrossV { get; set; }
        public float[,] CrossAttnOutput { get; set; }
        public float[,] CrossResidualInput { get; set; }
        public LayerNormCache LNCrossCache { get; set; }
        public float[,] NormedCross { get; set; }

        // Time decay
        public float[,] TimeDiffs { get; set; }
        public float[] KeyTimesFromRef { get; set; }
        public float[][,] CrossAttentionWeights { get; set; }
        public float[][,] CrossScoresPreSoftmax { get; set; }
        public ContentAwareDecayCache DecayCache { get; set; }

        // FFN
        public float[][] FFNInputRows { get; set; }
        public float[,] FFNOutput { get; set; }
        public float[,] FFNResidualInput { get; set; }
        public LayerNormCache LNFFNCache { get; set; }

        public BlockCache()
        {
            LNSelfCache = new LayerNormCache();
            LNCrossCache = new LayerNormCache();
            LNFFNCache = new LayerNormCache();
        }
        public void Reset()
        {
            BlockInput = null;

            // Self Attention
            SelfQ = null;
            SelfK = null;
            SelfV = null;
            SelfAttnOutput = null;
            SelfResidualInput = null;
            NormedSelf = null;

            // Cross Attention
            CrossQ = null;
            CrossK = null;
            CrossV = null;
            CrossAttnOutput = null;
            CrossResidualInput = null;
            NormedCross = null;

            // Time decay
            TimeDiffs = null;
            KeyTimesFromRef = null;
            CrossAttentionWeights = null;
            CrossScoresPreSoftmax = null;
            DecayCache = null;

            // FFN
            FFNInputRows = null;
            FFNOutput = null;
            FFNResidualInput = null;

            // Reset sub-caches (important)
            LNSelfCache?.Reset();
            LNCrossCache?.Reset();
            LNFFNCache?.Reset();
        }
    }

    public class CrossAttentionBlockGradients
    {
        public AttentionGradients SelfAttnGrads { get; set; }
        public LayerNormGradients LNSelfGrads { get; set; }
        public AttentionGradients CrossAttnGrads { get; set; }
        public LayerNormGradients LNCrossGrads { get; set; }
        public ContentAwareDecayGradients DecayGrads { get; set; }
        public LayerNormGradients LNFFNGrads { get; set; }

        public CrossAttentionBlockGradients(int embeddingDim, int numHeads, int decayProjectionDim, int decayHiddenDim, int decayTimeBases = 8)
        {
            SelfAttnGrads = new AttentionGradients(embeddingDim);
            LNSelfGrads = new LayerNormGradients(embeddingDim);
            CrossAttnGrads = new AttentionGradients(embeddingDim);
            LNCrossGrads = new LayerNormGradients(embeddingDim);
            LNFFNGrads = new LayerNormGradients(embeddingDim);

            if (numHeads > 0)
            {
                int mlpInputDim = decayProjectionDim * 3 + 2;
                int timeRawDim = 2 * decayTimeBases;
                DecayGrads = new ContentAwareDecayGradients(numHeads, embeddingDim, decayProjectionDim, decayHiddenDim, mlpInputDim, decayTimeBases, timeRawDim);
            }
        }

        public void Zero()
        {
            SelfAttnGrads.Zero();
            LNSelfGrads.Zero();
            CrossAttnGrads.Zero();
            LNCrossGrads.Zero();
            LNFFNGrads.Zero();
            DecayGrads?.Zero();
        }
    }

    public class MultimodalForwardCache
    {
        // Text Encoder
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
        public List<MultimodalForwardCache> StoryCaches { get; set; }
        public int[] StoryTokenCounts { get; set; }
        public float[] StoryArrivalTimes { get; set; }

        // Price Decoder
        public float[,] PriceEmbedded { get; set; }
        public float[,] PriceContinuousInput { get; set; }
        public List<BlockCache> PriceBlockCaches { get; set; }
        public float[,] PriceFinalHidden { get; set; }

        // Price context training metadata (set by ForwardWithPriceContextAndCache)
        public int NumNewsContext { get; set; } = 0;
        public int NumPriceContext { get; set; } = 0;
        public float[,] PriceContextHidden { get; set; }
        public void Reset()
        {
            // Text tensors
            TextEmbedded = null;
            TextTokenIds = null;
            TextFinalHidden = null;

            TextLayerInputs.Clear();
            TextFFNOutputs.Clear();
            TextFFNInputs.Clear();

            // Reset per-layer caches (already allocated in ctor)
            for (int i = 0; i < TextAttentionCaches.Count; i++)
            {
                TextAttentionCaches[i].Reset();
                TextLN1Caches[i].Reset();
                TextLN2Caches[i].Reset();
            }

            // Story data
            if (StoryCaches != null)
            {
                foreach (var storyCache in StoryCaches)
                {
                    storyCache?.Reset();  // FIX: Reset each sub-cache before discarding the list
                }
                StoryCaches.Clear();
            }
            StoryTokenCounts = null;
            StoryArrivalTimes = null;

            // Price tensors
            PriceEmbedded = null;
            PriceContinuousInput = null;
            PriceFinalHidden = null;
            PriceContextHidden = null;

            NumNewsContext = 0;
            NumPriceContext = 0;

            // Reset block caches (already allocated)
            for (int i = 0; i < PriceBlockCaches.Count; i++)
            {
                PriceBlockCaches[i].Reset();
            }
        }
        public MultimodalForwardCache(int textNumLayers, int priceNumLayers)
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

    }
}
