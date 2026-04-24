using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.CrossAttentionMultimodal
{
    /// <summary>
    /// Top-level configuration for CrossAttentionMultimodal (Flamingo-style).
    /// Composes: <see cref="TextEncoderConfig"/>, <see cref="PriceDecoderConfig"/>,
    /// <see cref="OutputHeadConfig"/>, <see cref="RuntimeConfig"/>,
    /// <see cref="RegularizationConfig"/>, and optionally
    /// <see cref="DecayNetworkConfig"/> / <see cref="PriceContextConfig"/>.
    /// </summary>
    public sealed class MultimodalTransformerConfig : ConfigBase
    {
        public TextEncoderConfig Text { get; set; } = new TextEncoderConfig();
        public PriceDecoderConfig Price { get; set; } = new PriceDecoderConfig();
        public OutputHeadConfig Output { get; set; } = new OutputHeadConfig();
        public RuntimeConfig Runtime { get; set; } = new RuntimeConfig();
        public RegularizationConfig Regularization { get; set; } = new RegularizationConfig();

        /// <summary>Optional; set Enabled = true for TACAMT-style decay.</summary>
        public DecayNetworkConfig DecayNetwork { get; set; } = new DecayNetworkConfig();

        /// <summary>Optional; set Enabled = true to use price-context training splits.</summary>
        public PriceContextConfig PriceContext { get; set; } = new PriceContextConfig();

        /// <summary>Optional; configure memory-bank pruning for TACAMT-style models.</summary>
        public MemoryPruningConfig MemoryPruning { get; set; } = new MemoryPruningConfig();

        /// <summary>
        /// Enforces Text.EmbeddingDim == Price.EmbeddingDim at validation time.
        /// Should always be true for cross-attention.
        /// </summary>
        public bool RequireSharedCrossAttentionEmbeddingDim { get; set; } = true;
        public override void Validate()
        {
            if (Text == null) throw new ArgumentNullException(nameof(Text));
            if (Price == null) throw new ArgumentNullException(nameof(Price));
            if (Output == null) throw new ArgumentNullException(nameof(Output));
            if (Runtime == null) throw new ArgumentNullException(nameof(Runtime));
            if (Regularization == null) throw new ArgumentNullException(nameof(Regularization));
            if (DecayNetwork == null) throw new ArgumentNullException(nameof(DecayNetwork));
            if (PriceContext == null) throw new ArgumentNullException(nameof(PriceContext));
            if (MemoryPruning == null) throw new ArgumentNullException(nameof(MemoryPruning));

            Text.Validate();
            Price.Validate();
            Output.Validate();
            Runtime.Validate();
            Regularization.Validate();
            DecayNetwork.Validate();
            PriceContext.Validate();
            MemoryPruning.Validate();

            // This implementation has no adapter/projection bridge between text-hidden dim
            // and price-hidden dim, so cross-attention needs them equal even if the flag is false.
            if (RequireSharedCrossAttentionEmbeddingDim || Text.EmbeddingDim != Price.EmbeddingDim)
            {
                Require(
                    Text.EmbeddingDim == Price.EmbeddingDim,
                    $"For this cross-attention implementation, {nameof(Text)}.{nameof(Text.EmbeddingDim)} ({Text.EmbeddingDim}) " +
                    $"must equal {nameof(Price)}.{nameof(Price.EmbeddingDim)} ({Price.EmbeddingDim}).");
            }
        }
    }
}
