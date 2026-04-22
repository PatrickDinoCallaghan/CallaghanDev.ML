using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MMTAC
{
    /// <summary>
    /// MMTAC output head and per-head loss weights.
    /// MMTAC decomposes prediction into High/Low/Close regression plus
    /// Range, Quality, Direction, and MidDirection heads.
    /// </summary>
    public sealed class MmtacOutputConfig : ConfigBase
    {
        /// <summary>Regression outputs: High, Low, Close.</summary>
        public const int RegressionOutputCount = 3;

        public bool UseConfidenceHead { get; set; } = false;

        public float DirectionLossWeight { get; set; } = 1.0f;
        public float MidDirectionLossWeight { get; set; } = 0.5f;
        public float RangeLossWeight { get; set; } = 1.0f;
        public float QualityLossWeight { get; set; } = 1.0f;

        /// <summary>
        /// Auxiliary loss that encourages predicted Close to move in the same
        /// direction as the Direction target relative to the previous observed close.
        /// </summary>
        public float CloseDirectionConsistencyWeight { get; set; } = 1.0f;

        /// <summary>
        /// Optional margin for the close-direction consistency loss.
        /// A positive margin requires predicted Close to move at least this far
        /// in the target direction relative to the previous close.
        /// </summary>
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