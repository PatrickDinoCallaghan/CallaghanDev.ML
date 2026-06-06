using System.Collections.Generic;
using CallaghanDev.ML.Transformers.TACAMT;

namespace CallaghanDev.ML.Transformers.PriceTAC
{
    public sealed class PriceTacForwardCache
    {
        // Global token
        public float[] GlobalTokenEmbedded { get; set; }
        public float[] GlobalRawInput { get; set; }

        // Price decoder
        public float[,] PriceEmbedded { get; set; }
        public float[,] PriceContinuousInput { get; set; }
        public int PriceContinuousInputRowStart { get; set; } = 0;
        public int PriceContinuousInputRowCount { get; set; } = 0;
        public List<BlockCache> PriceBlockCaches { get; }
        public float[,] PriceFinalHidden { get; set; }

        // Combined context metadata: [global?] [price context]
        public int NumPriceContext { get; set; } = 0;
        public int NumGlobalContext { get; set; } = 0;
        public float[,] PriceContextHidden { get; set; }
        public float[,] ContextHidden { get; set; }
        public float[] ContextTimes { get; set; }

        // Output-head logits needed for backward
        public float[] RangeLogits { get; set; }
        public float[] QualityLogits { get; set; }
        public float[,] RegressionLogits { get; set; }

        public PriceTacForwardCache(int priceNumLayers)
        {
            PriceBlockCaches = new List<BlockCache>(priceNumLayers);
            for (int i = 0; i < priceNumLayers; i++)
                PriceBlockCaches.Add(new BlockCache());
        }

        public void Reset()
        {
            GlobalTokenEmbedded = null;
            GlobalRawInput = null;

            RegressionLogits = null;
            RangeLogits = null;
            QualityLogits = null;

            PriceEmbedded = null;
            PriceContinuousInput = null;
            PriceContinuousInputRowStart = 0;
            PriceContinuousInputRowCount = 0;
            PriceFinalHidden = null;
            PriceContextHidden = null;
            ContextHidden = null;
            ContextTimes = null;

            NumPriceContext = 0;
            NumGlobalContext = 0;

            for (int i = 0; i < PriceBlockCaches.Count; i++)
                PriceBlockCaches[i].Reset();
        }
    }
}
