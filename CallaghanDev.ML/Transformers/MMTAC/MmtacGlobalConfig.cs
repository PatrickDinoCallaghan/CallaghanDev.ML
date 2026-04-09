using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MMTAC
{
    /// <summary>
    /// MMTAC global-feature token settings.
    /// Controls the optional single global-context vector prepended to price decoder input.
    /// </summary>
    public sealed class MmtacGlobalConfig : ConfigBase
    {
        /// <summary>
        /// Dimension of the external global feature vector (e.g. macro indicators).
        /// Set to 0 to disable the global token entirely.
        /// </summary>
        public int GlobalFeatureDim { get; set; } = 0;

        /// <summary>
        /// When true, the global token bypasses the decay network and is injected
        /// directly into cross-attention as a permanent context entry.
        /// </summary>
        public bool BypassDecay { get; set; } = true;

        public override void Validate()
        {
            if (GlobalFeatureDim < 0)
                throw new ArgumentException($"{nameof(GlobalFeatureDim)} must be >= 0, got {GlobalFeatureDim}.");
        }
    }
}
