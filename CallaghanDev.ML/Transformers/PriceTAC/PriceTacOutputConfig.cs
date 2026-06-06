using CallaghanDev.ML.Transformers.Configuration;

namespace CallaghanDev.ML.Transformers.PriceTAC
{
    /// <summary>
    /// PriceTAC output head and per-head loss weights.  Matches MMTAC's price
    /// output contract: High/Low/Close plus Range, Quality, Direction, MidDirection,
    /// and optional Confidence.
    /// </summary>
    public sealed class PriceTacOutputConfig : ConfigBase
    {
        public const int RegressionOutputCount = 3;

        public bool UseConfidenceHead { get; set; } = false;

        public float DirectionLossWeight { get; set; } = 1.0f;
        public float MidDirectionLossWeight { get; set; } = 0.5f;
        public float RangeLossWeight { get; set; } = 1.0f;
        public float QualityLossWeight { get; set; } = 1.0f;

        public float CloseDirectionConsistencyWeight { get; set; } = 1.0f;
        public float CloseDirectionConsistencyMargin { get; set; } = 0.02f;

        public override void Validate()
        {
            RequireNonNegative(DirectionLossWeight, nameof(DirectionLossWeight));
            RequireNonNegative(MidDirectionLossWeight, nameof(MidDirectionLossWeight));
            RequireNonNegative(RangeLossWeight, nameof(RangeLossWeight));
            RequireNonNegative(QualityLossWeight, nameof(QualityLossWeight));
            RequireNonNegative(CloseDirectionConsistencyWeight, nameof(CloseDirectionConsistencyWeight));
            RequireNonNegative(CloseDirectionConsistencyMargin, nameof(CloseDirectionConsistencyMargin));
        }
    }
}
