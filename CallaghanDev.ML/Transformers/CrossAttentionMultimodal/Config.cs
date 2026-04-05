using CallaghanDev.ML.Enums;
using System;

namespace CallaghanDev.ML.Transformers.CrossAttentionMultimodal
{


    /// <summary>
    /// Configuration for the cross-attention multimodal transformer.
    /// 
    /// Architecture (Flamingo-style):
    ///   Text Encoder:  Standard transformer (Text/SymbolicSequence mode) - frozen or trainable
    ///   Price Decoder:  Transformer with interleaved cross-attention layers attending to text encoder output
    ///   Output Head:    Linear projection from price decoder hidden states => OHLCV predictions + confidence
    /// </summary>
    public class Config
    {
        // =====================================================================
        // TEXT ENCODER CONFIG
        // =====================================================================

        /// <summary>Vocabulary size for the text encoder's token embedding.</summary>
        public int TextVocabSize { get; set; } = 5000;

        /// <summary>Maximum text sequence length.</summary>
        public int TextMaxSequenceLength { get; set; } = 256;

        /// <summary>Embedding dimension for the text encoder.</summary>
        public int TextEmbeddingDim { get; set; } = 128;

        /// <summary>Number of attention heads in the text encoder.</summary>
        public int TextNumHeads { get; set; } = 4;

        /// <summary>Number of transformer layers in the text encoder.</summary>
        public int TextNumLayers { get; set; } = 4;

        /// <summary>FFN hidden dimension in the text encoder.</summary>
        public int TextFeedForwardDim { get; set; } = 512;

        /// <summary>
        /// Whether to freeze the text encoder during multimodal training.
        /// If true, only the price decoder and cross-attention layers are trained.
        /// Set to false to fine-tune the text encoder end-to-end.
        /// </summary>
        public bool FreezeTextEncoder { get; set; } = false;

        /// <summary>
        /// Whether the text encoder uses causal (decoder-only) masking.
        /// For encoding context, bidirectional (false) is typically better.
        /// </summary>
        public bool TextUseDecoderOnly { get; set; } = false;

        // =====================================================================
        // PRICE DECODER CONFIG (with cross-attention)
        // =====================================================================

        /// <summary>Number of input features per price timestep (e.g., OHLCV = 5).</summary>
        public int PriceInputFeatureDim { get; set; } = 5;

        /// <summary>Maximum price sequence length.</summary>
        public int PriceMaxSequenceLength { get; set; } = 128;

        /// <summary>Embedding dimension for the price decoder. Must match TextEmbeddingDim for cross-attention.</summary>
        public int PriceEmbeddingDim { get; set; } = 128;

        /// <summary>Number of attention heads in the price decoder (self-attention and cross-attention).</summary>
        public int PriceNumHeads { get; set; } = 4;

        /// <summary>Number of transformer layers in the price decoder (each has self-attn + cross-attn + FFN).</summary>
        public int PriceNumLayers { get; set; } = 4;

        /// <summary>FFN hidden dimension in the price decoder.</summary>
        public int PriceFeedForwardDim { get; set; } = 512;

        /// <summary>Whether the price decoder uses causal masking for self-attention.</summary>
        public bool PriceUseDecoderOnly { get; set; } = true;

        // =====================================================================
        // OUTPUT HEAD CONFIG
        // =====================================================================

        /// <summary>
        /// Number of output values to predict per timestep.
        /// E.g., 5 for OHLCV, or 6 for OHLCV + confidence score.
        /// </summary>
        public int OutputDim { get; set; } = 5;

        /// <summary>
        /// Whether to include a confidence head (sigmoid output) in addition to OHLCV predictions.
        /// If true, the model outputs OutputDim regression values + 1 confidence value.
        /// </summary>
        public bool UseConfidenceHead { get; set; } = true;

        /// <summary>
        /// The total effective output dimension including confidence if enabled.
        /// </summary>
        public int EffectiveOutputDim => UseConfidenceHead ? OutputDim + 1 : OutputDim;

        // =====================================================================
        // SHARED / HARDWARE CONFIG
        // =====================================================================

        public ActivationType FFNActivationType { get; set; } = ActivationType.Relu;
        public AccelerationType AccelerationType { get; set; } = AccelerationType.CPU;
        public int AccelerationDeviceId { get; set; } = 0;
        public float L2RegulationLamda { get; set; } = 0.01f;
        public float GradientClippingThreshold { get; set; } = 1.0f;

        // =====================================================================
        // VALIDATION
        // =====================================================================

        public void Validate()
        {
            if (PriceEmbeddingDim != TextEmbeddingDim)
                throw new ArgumentException(
                    $"PriceEmbeddingDim ({PriceEmbeddingDim}) must equal TextEmbeddingDim ({TextEmbeddingDim}) " +
                    "for cross-attention to work. The query (price) and key/value (text) must share the same dimension.");

            if (TextEmbeddingDim % TextNumHeads != 0)
                throw new ArgumentException(
                    $"TextEmbeddingDim ({TextEmbeddingDim}) must be divisible by TextNumHeads ({TextNumHeads}).");

            if (PriceEmbeddingDim % PriceNumHeads != 0)
                throw new ArgumentException(
                    $"PriceEmbeddingDim ({PriceEmbeddingDim}) must be divisible by PriceNumHeads ({PriceNumHeads}).");
        }
    }
}
