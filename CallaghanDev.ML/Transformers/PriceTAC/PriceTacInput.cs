using System;

namespace CallaghanDev.ML.Transformers.PriceTAC
{
    /// <summary>
    /// Price-only input for PriceTAC.  The model consumes price rows plus an optional
    /// global feature vector; there is intentionally no language-side payload.
    /// </summary>
    public sealed class PriceTacInput
    {
        public DateTime PredictionTimestamp { get; set; }
        public float[,] PriceSequence { get; set; }
        public float[] GlobalFeatures { get; set; }
    }
}
