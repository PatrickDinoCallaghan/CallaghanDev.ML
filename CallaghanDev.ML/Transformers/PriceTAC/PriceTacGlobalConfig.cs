using CallaghanDev.ML.Transformers.Configuration;
using System;

namespace CallaghanDev.ML.Transformers.PriceTAC
{
    /// <summary>
    /// Optional global-feature token settings.  When enabled, the global token is
    /// prepended to the price-memory context and can bypass learned time decay.
    /// </summary>
    public sealed class PriceTacGlobalConfig : ConfigBase
    {
        public int GlobalFeatureDim { get; set; } = 0;
        public bool BypassDecay { get; set; } = true;

        public override void Validate()
        {
            if (GlobalFeatureDim < 0)
                throw new ArgumentException($"{nameof(GlobalFeatureDim)} must be >= 0, got {GlobalFeatureDim}.");
        }
    }
}
