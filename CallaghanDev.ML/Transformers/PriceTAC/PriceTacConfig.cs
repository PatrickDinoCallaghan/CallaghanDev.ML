using CallaghanDev.ML.Transformers.Configuration;
using System;

namespace CallaghanDev.ML.Transformers.PriceTAC
{
    /// <summary>
    /// Top-level configuration for the price-only MMTAC variant.
    /// It keeps MMTAC's price decoder, global-feature token, learned content-aware
    /// time decay, output heads, price-context training, and rolling price memory,
    /// but has no language-side inputs.
    /// </summary>
    public sealed class PriceTacConfig : ConfigBase
    {
        public PriceDecoderConfig Price { get; set; } = new PriceDecoderConfig();
        public PriceTacGlobalConfig Global { get; set; } = new PriceTacGlobalConfig();
        public PriceTacOutputConfig Output { get; set; } = new PriceTacOutputConfig();
        public DecayNetworkConfig Decay { get; set; } = new DecayNetworkConfig { Enabled = true };
        public RegularizationConfig Reg { get; set; } = new RegularizationConfig();
        public RuntimeConfig Runtime { get; set; } = new RuntimeConfig();
        public PriceContextConfig PriceContext { get; set; } = new PriceContextConfig { Enabled = true };
        public MemoryPruningConfig Pruning { get; set; } = new MemoryPruningConfig();

        /// <summary>
        /// Shape kept at three rows for compatibility with existing acceleration helpers:
        /// row 0 is unused, row 1 = price-memory, row 2 = global.
        /// </summary>
        public const int ContextTypeCount = 3;

        public override void Validate()
        {
            if (Price == null) throw new ArgumentNullException(nameof(Price));
            if (Global == null) throw new ArgumentNullException(nameof(Global));
            if (Output == null) throw new ArgumentNullException(nameof(Output));
            if (Decay == null) throw new ArgumentNullException(nameof(Decay));
            if (Reg == null) throw new ArgumentNullException(nameof(Reg));
            if (Runtime == null) throw new ArgumentNullException(nameof(Runtime));
            if (PriceContext == null) throw new ArgumentNullException(nameof(PriceContext));
            if (Pruning == null) throw new ArgumentNullException(nameof(Pruning));

            Price.Validate();
            Global.Validate();
            Output.Validate();
            Decay.Validate();
            Reg.Validate();
            Runtime.Validate();
            PriceContext.Validate();
            Pruning.Validate();

            if (Global.GlobalFeatureDim > 0)
                RequirePositive(Price.EmbeddingDim, $"{nameof(Price)}.{nameof(Price.EmbeddingDim)}");
        }

        public static PriceTacConfig Small(int priceFeatureDim = 5) =>
            new PriceTacConfig
            {
                Price = new PriceDecoderConfig { InputFeatureDim = priceFeatureDim, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 128 },
                Global = new PriceTacGlobalConfig { GlobalFeatureDim = 0 },
            };

        public static PriceTacConfig Standard(int priceFeatureDim = 5, int globalDim = 16) =>
            new PriceTacConfig
            {
                Price = new PriceDecoderConfig { InputFeatureDim = priceFeatureDim, EmbeddingDim = 128, NumHeads = 4, NumLayers = 3, FeedForwardDim = 256 },
                Global = new PriceTacGlobalConfig { GlobalFeatureDim = globalDim, BypassDecay = true },
            };
    }
}
