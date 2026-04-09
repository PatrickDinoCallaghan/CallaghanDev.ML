using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MMTAC
{

    /// <summary>
    /// Top-level configuration for the MMTAC architecture.
    /// Extends the multimodal base with a global-feature token and
    /// MMTAC-specific output loss weights.
    /// </summary>
    public sealed class MmtacConfig : ConfigBase
    {
        public TextEncoderConfig Text { get; set; } = new TextEncoderConfig();
        public PriceDecoderConfig Price { get; set; } = new PriceDecoderConfig();
        public MmtacGlobalConfig Global { get; set; } = new MmtacGlobalConfig();
        public MmtacOutputConfig Output { get; set; } = new MmtacOutputConfig();
        public DecayNetworkConfig Decay { get; set; } = new DecayNetworkConfig { Enabled = true };
        public RegularizationConfig Reg { get; set; } = new RegularizationConfig();
        public RuntimeConfig Runtime { get; set; } = new RuntimeConfig();
        public PriceContextConfig PriceContext { get; set; } = new PriceContextConfig { Enabled = true };
        public MemoryPruningConfig Pruning { get; set; } = new MemoryPruningConfig();

        /// <summary>
        /// Number of learnable context-type embeddings (news, price-memory, …).
        /// Baked into model weight shapes — do not change after initialisation.
        /// </summary>
        public const int ContextTypeCount = 3;

        public override void Validate()
        {
            if (Text == null) throw new ArgumentNullException(nameof(Text));
            if (Price == null) throw new ArgumentNullException(nameof(Price));
            if (Global == null) throw new ArgumentNullException(nameof(Global));
            if (Output == null) throw new ArgumentNullException(nameof(Output));
            if (Decay == null) throw new ArgumentNullException(nameof(Decay));
            if (Reg == null) throw new ArgumentNullException(nameof(Reg));
            if (Runtime == null) throw new ArgumentNullException(nameof(Runtime));
            if (PriceContext == null) throw new ArgumentNullException(nameof(PriceContext));
            if (Pruning == null) throw new ArgumentNullException(nameof(Pruning));

            Text.Validate();
            Price.Validate();
            Global.Validate();
            Output.Validate();
            Decay.Validate();
            Reg.Validate();
            Runtime.Validate();
            PriceContext.Validate();
            Pruning.Validate();

            Require(Text.EmbeddingDim == Price.EmbeddingDim, $"Text.EmbeddingDim ({Text.EmbeddingDim}) must equal Price.EmbeddingDim ({Price.EmbeddingDim}).");

            if (Global.GlobalFeatureDim > 0)
                RequirePositive(Price.EmbeddingDim, $"{nameof(Price)}.{nameof(Price.EmbeddingDim)}");
        }

        //  Preset factories 

        public static MmtacConfig Small(int vocabSize = 10000, int priceFeatureDim = 5) =>
            new MmtacConfig
            {
                Text = new TextEncoderConfig { VocabSize = vocabSize, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 128 },
                Price = new PriceDecoderConfig { InputFeatureDim = priceFeatureDim, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 128 },
                Global = new MmtacGlobalConfig { GlobalFeatureDim = 0 },
            };

        public static MmtacConfig Standard(int vocabSize = 50000, int priceFeatureDim = 5, int globalDim = 16) =>
            new MmtacConfig
            {
                Text = new TextEncoderConfig { VocabSize = vocabSize, EmbeddingDim = 128, NumHeads = 4, NumLayers = 4, FeedForwardDim = 256 },
                Price = new PriceDecoderConfig { InputFeatureDim = priceFeatureDim, EmbeddingDim = 128, NumHeads = 4, NumLayers = 3, FeedForwardDim = 256 },
                Global = new MmtacGlobalConfig { GlobalFeatureDim = globalDim, BypassDecay = true },
            };
    }
}
