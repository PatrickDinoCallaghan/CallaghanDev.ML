using CallaghanDev.ML.Enums;
using System;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public class Config
    {
        #region Text encoder

        /// <summary>
        /// Size of the text vocabulary (number of unique tokens).
        /// Must match the tokenizer's vocab size if one is attached.
        /// </summary>
        public int TextVocabSize { get; set; } = 50000;

        /// <summary>
        /// Maximum sequence length for text inputs.
        /// Determines the size of positional encoding tables.
        /// </summary>
        public int TextMaxSequenceLength { get; set; } = 1024;//= 2048;

        /// <summary>
        /// Embedding dimension for text tokens.
        /// Must be divisible by TextNumHeads.
        /// </summary>
        public int TextEmbeddingDim { get; set; } = 128;

        /// <summary>
        /// Number of attention heads in the text encoder.
        /// TextEmbeddingDim must be divisible by this value.
        /// </summary>
        public int TextNumHeads { get; set; } = 4;

        /// <summary>
        /// Number of transformer layers in the text encoder.
        /// </summary>
        public int TextNumLayers { get; set; } = 4;

        /// <summary>
        /// Feed-forward hidden dimension in the text encoder blocks.
        /// </summary>
        public int TextFeedForwardDim { get; set; } = 256;

        /// <summary>
        /// Whether the text encoder uses causal (decoder-only) masking.
        /// If false, the text encoder is bidirectional (standard encoder).
        /// </summary>
        public bool TextUseDecoderOnly { get; set; } = false;

        /// <summary>
        /// Whether to freeze the text encoder during training.
        /// When true, no gradients flow back through text encoder parameters.
        /// </summary>
        public bool FreezeTextEncoder { get; set; } = false;

        #endregion

        #region Price decoder

        /// <summary>
        /// Number of input features per price timestep (e.g. OHLCV = 5).
        /// </summary>
        public int PriceInputFeatureDim { get; set; } = 5;

        /// <summary>
        /// Maximum sequence length for price inputs.
        /// Determines the size of positional encoding tables.
        /// </summary>
        public int PriceMaxSequenceLength { get; set; } = 256;

        /// <summary>
        /// Embedding dimension for the price decoder.
        /// Must be divisible by PriceNumHeads.
        /// </summary>
        public int PriceEmbeddingDim { get; set; } = 128;

        /// <summary>
        /// Number of attention heads in the price decoder.
        /// PriceEmbeddingDim must be divisible by this value.
        /// </summary>
        public int PriceNumHeads { get; set; } = 4;

        /// <summary>
        /// Number of cross-attention block layers in the price decoder.
        /// </summary>
        public int PriceNumLayers { get; set; } = 3;

        /// <summary>
        /// Feed-forward hidden dimension in the price decoder blocks.
        /// </summary>
        public int PriceFeedForwardDim { get; set; } = 256;

        /// <summary>
        /// Whether the price decoder uses causal (decoder-only) masking
        /// for its self-attention. Typically true so each price position
        /// can only attend to previous positions.
        /// </summary>
        public bool PriceUseDecoderOnly { get; set; } = true;

        #endregion

        #region Output heads

        /// <summary>
        /// Output dimension per timestep (e.g. 1 for next-close prediction,
        /// 5 for OHLCV prediction).
        /// </summary>
        public int OutputDim { get; set; } = 1;

        /// <summary>
        /// Whether to include a confidence head that outputs a sigmoid
        /// score indicating the model's confidence in each prediction.
        /// </summary>
        public bool UseConfidenceHead { get; set; } = false;

        #endregion

        #region Activation and acceleration

        /// <summary>
        /// Activation function type for feed-forward networks.
        /// </summary>
        public ActivationType FFNActivationType { get; set; } = ActivationType.Relu;

        /// <summary>
        /// Hardware acceleration type (CPU, MultiThreadCPU, GPU, CUDA).
        /// </summary>
        public AccelerationType AccelerationType { get; set; } = AccelerationType.MultiThreadCPU;

        /// <summary>
        /// Device ID for GPU/CUDA acceleration.
        /// </summary>
        public int AccelerationDeviceId { get; set; } = 0;

        #endregion

        #region Regularization

        /// <summary>
        /// L2 regularization lambda for attention and FFN weights.
        /// </summary>
        public float L2RegulationLamda { get; set; } = 0.01f;

        /// <summary>
        /// Global gradient clipping threshold. Applied after accumulating
        /// all gradients for a batch.
        /// </summary>
        public float GradientClippingThreshold { get; set; } = 0.5f;

        #endregion

        #region Content aware decay network

        /// <summary>
        /// Projection dimension for the content-aware decay network.
        /// Both query (current price) and key (memory) embeddings are projected
        /// to this dimension per head before computing decay gates.
        /// Higher = more expressive but more parameters. Default: 8
        /// </summary>
        public int DecayProjectionDim { get; set; } = 16;

        /// <summary>
        /// Hidden dimension for the content-aware decay MLP.
        /// The gating MLP is: (projDim*3 + 2) -> hiddenDim -> 1.
        /// Default: 16
        /// </summary>
        public int DecayHiddenDim { get; set; } = 32;

        /// <summary>
        /// Number of sinusoidal basis functions per head for multi-scale time encoding.
        /// Each basis has a learned log-frequency, producing sin and cos components.
        /// Total time encoding features per key = numBases * 2, which are linearly
        /// projected to projectionDim. This captures temporal patterns at multiple
        /// scales (seconds, minutes, hours, days) simultaneously.
        /// Must be >= 1. Default: 8
        /// </summary>
        public int DecayTimeEncodingBases { get; set; } = 16;

        /// <summary>
        /// Dropout rate for memory attention weights in the decay network.
        /// Applied during training only. Prevents the memory interaction attention
        /// from overfitting to specific memory entry patterns.
        /// Default: 0.1
        /// </summary>
        public float DecayMemAttnDropout { get; set; } = 0.2f;

        /// <summary>
        /// Dropout rate for the MLP hidden layer in the decay network.
        /// Applied during training only. Regularizes the interaction term
        /// (query_proj ⊙ refined_key) to prevent overfitting to spurious
        /// correlations between current prices and historical content.
        /// Default: 0.1
        /// </summary>
        public float DecayMLPDropout { get; set; } = 0.2f;

        /// <summary>
        /// L2 weight decay coefficient for all decay network parameters.
        /// Added as grad += weightDecay * weight during backpropagation.
        /// Default: 0.0 (disabled). Recommended range: 1e-5 to 1e-3.
        /// </summary>
        public float DecayWeightDecay { get; set; } = 0.0f;

        #endregion

        #region price context training

        /// <summary>
        /// Minimum length of the history portion when splitting price sequences
        /// for price context training. Sequences shorter than
        /// (PriceContextMinHistoryLength + PriceContextMinCurrentLength + 1)
        /// cannot be split and train without price context.
        /// Default: 5
        /// </summary>
        public int PriceContextMinHistoryLength { get; set; } = 5;

        /// <summary>
        /// Minimum length of the current portion when splitting.
        /// Default: 5
        /// </summary>
        public int PriceContextMinCurrentLength { get; set; } = 5;

        #endregion

        #region Validation

        /// <summary>
        /// Validates the configuration, ensuring dimensional consistency.
        /// Throws ArgumentException if any constraint is violated.
        /// </summary>
        public void Validate()
        {
            if (TextEmbeddingDim % TextNumHeads != 0)
            {
                throw new ArgumentException($"TextEmbeddingDim ({TextEmbeddingDim}) must be divisible by TextNumHeads ({TextNumHeads}).");
            }
            if (PriceEmbeddingDim % PriceNumHeads != 0)
            {
                throw new ArgumentException($"PriceEmbeddingDim ({PriceEmbeddingDim}) must be divisible by PriceNumHeads ({PriceNumHeads}).");
            }
            if (TextVocabSize <= 0)
            {
                throw new ArgumentException($"TextVocabSize must be positive, got {TextVocabSize}.");
            }
            if (TextMaxSequenceLength <= 0)
            {
                throw new ArgumentException($"TextMaxSequenceLength must be positive, got {TextMaxSequenceLength}.");
            }
            if (PriceMaxSequenceLength <= 0)
            {
                throw new ArgumentException($"PriceMaxSequenceLength must be positive, got {PriceMaxSequenceLength}.");
            }
            if (PriceInputFeatureDim <= 0)
            {
                throw new ArgumentException($"PriceInputFeatureDim must be positive, got {PriceInputFeatureDim}.");
            }
            if (OutputDim <= 0)
            {
                throw new ArgumentException($"OutputDim must be positive, got {OutputDim}.");
            }
            if (TextNumLayers <= 0)
            {
                throw new ArgumentException($"TextNumLayers must be positive, got {TextNumLayers}.");
            }
            if (PriceNumLayers <= 0)
            {
                throw new ArgumentException($"PriceNumLayers must be positive, got {PriceNumLayers}.");
            }

            if (DecayProjectionDim <= 0)
            {
                throw new ArgumentException($"DecayProjectionDim must be positive, got {DecayProjectionDim}.");
            }

            if (DecayHiddenDim <= 0)
            {
                throw new ArgumentException($"DecayHiddenDim must be positive, got {DecayHiddenDim}.");
            }

            if (DecayTimeEncodingBases <= 0)
            {
                throw new ArgumentException($"DecayTimeEncodingBases must be positive, got {DecayTimeEncodingBases}.");
            }

            if (DecayMemAttnDropout < 0f || DecayMemAttnDropout >= 1f)
            {
                throw new ArgumentException($"DecayMemAttnDropout must be in [0, 1), got {DecayMemAttnDropout}.");
            }

            if (DecayMLPDropout < 0f || DecayMLPDropout >= 1f)
            {
                throw new ArgumentException($"DecayMLPDropout must be in [0, 1), got {DecayMLPDropout}.");
            }

            if (DecayWeightDecay < 0f)
            {
                throw new ArgumentException($"DecayWeightDecay must be non-negative, got {DecayWeightDecay}.");
            }

            if (L2RegulationLamda < 0f)
            {
                throw new ArgumentException($"L2RegulationLamda must be non-negative, got {L2RegulationLamda}.");
            }

            if (GradientClippingThreshold <= 0f)
            {
                throw new ArgumentException($"GradientClippingThreshold must be positive, got {GradientClippingThreshold}.");
            }

            if (PriceContextMinHistoryLength <= 0)
            {
                throw new ArgumentException($"PriceContextMinHistoryLength must be positive, got {PriceContextMinHistoryLength}.");
            }

            if (PriceContextMinCurrentLength <= 0)
            {
                throw new ArgumentException($"PriceContextMinCurrentLength must be positive, got {PriceContextMinCurrentLength}.");
            }
        }

        #endregion
    }
}