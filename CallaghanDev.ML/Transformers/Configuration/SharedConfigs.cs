using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.Configuration
{
    /// <summary>
    /// Unified training-loop configuration for all transformer models.
    /// <para>
    /// Architecture notes:
    /// <list type="bullet">
    ///   <item>MultiTypeTransformer  - uses all fields; ConfidenceLossWeight unused (leave 0f)</item>
    ///   <item>CrossAttentionMultimodal - uses all fields; WarmupEpochs unused (leave 0)</item>
    ///   <item>TACAMT                - uses all fields including WarmupEpochs</item>
    ///   <item>MMTAC                 - uses all fields including WarmupEpochs</item>
    /// </list>
    /// </para>
    /// </summary>
    public sealed class TrainingConfig : ConfigBase
    {
        /// <summary>
        /// Initial learning rate. Set to 0f to let the trainer auto-calculate
        /// from epoch count (CrossAttentionMultimodal / TACAMT behaviour).
        /// </summary>
        public float LearningRate { get; set; } = 0.0001f;

        public int BatchSize { get; set; } = 32;

        public int Epochs { get; set; } = 10;

        public int ValidationInterval { get; set; } = 100;

        public bool Verbose { get; set; } = true;

        //  Gradient clipping 

        public bool UseGradientClipping { get; set; } = true;

        public float GradientClipThreshold { get; set; } = 1.0f;

        //  Learning-rate schedule 

        public bool UseLearningRateDecay { get; set; } = false;

        /// <summary>Multiplicative decay per epoch. Must be in (0, 1).</summary>
        public float LearningRateDecay { get; set; } = 0.95f;

        //  Warm-up (TACAMT / MMTAC) 

        /// <summary>
        /// Linear LR warm-up epochs. LR ramps from 0 → LearningRate over this
        /// many epochs. Set to 0 to disable (default for non-MMTAC architectures).
        /// </summary>
        public int WarmupEpochs { get; set; } = 0;

        //  Confidence head 

        /// <summary>
        /// Weight on the BCE confidence loss term. Only meaningful when the
        /// model has UseConfidenceHead = true. Leave at 0f otherwise.
        /// </summary>
        public float ConfidenceLossWeight { get; set; } = 1.0f;



        //  Validation 

        public override void Validate()
        {
            RequireNonNegative(LearningRate, nameof(LearningRate));
            RequirePositive(BatchSize, nameof(BatchSize));
            RequirePositive(Epochs, nameof(Epochs));
            RequirePositive(ValidationInterval, nameof(ValidationInterval));

            if (WarmupEpochs < 0)
                throw new ArgumentException($"{nameof(WarmupEpochs)} must be >= 0, got {WarmupEpochs}.");

            if (UseGradientClipping)
                RequirePositive(GradientClipThreshold, nameof(GradientClipThreshold));

            if (UseLearningRateDecay)
                RequireInRangeZeroToOneExclusiveUpper(LearningRateDecay, nameof(LearningRateDecay));

            if (ConfidenceLossWeight != 0f)
                RequireNonNegative(ConfidenceLossWeight, nameof(ConfidenceLossWeight));
        }

        //  Preset factories 

        /// <summary>Defaults for MultiTypeTransformer (single-modal).</summary>
        public static TrainingConfig ForLanguageModel() => new TrainingConfig
        {
            LearningRate = 0.0001f,
            BatchSize = 32,
            Epochs = 10,
            ValidationInterval = 100,
            UseGradientClipping = true,
            GradientClipThreshold = 1.0f,
            WarmupEpochs = 0,
        };

        /// <summary>Defaults for CrossAttentionMultimodal.</summary>
        public static TrainingConfig ForMultimodalModel() => new TrainingConfig
        {
            LearningRate = 0f,     // trainer auto-calculates
            BatchSize = 4,
            Epochs = 50,
            ValidationInterval = 10,
            UseGradientClipping = true,
            GradientClipThreshold = 1.0f,
            ConfidenceLossWeight = 0.1f,
            WarmupEpochs = 0,
        };

        /// <summary>Defaults for TACAMT / MMTAC (multimodal with warm-up).</summary>
        public static TrainingConfig ForMmtac() => new TrainingConfig
        {
            LearningRate = 0.0001f,
            BatchSize = 8,
            Epochs = 50,
            ValidationInterval = 10,
            UseGradientClipping = true,
            GradientClipThreshold = 1.0f,
            ConfidenceLossWeight = 0.1f,
            WarmupEpochs = 3,
        };
    }

    /// <summary>
    /// Common transformer-block shape shared by all encoder and decoder modules
    /// across every architecture.
    /// </summary>
    public abstract class TransformerModuleConfigBase : ConfigBase
    {
        /// <summary>Embedding dimension. Must be divisible by NumHeads.</summary>
        public int EmbeddingDim { get; set; } = 128;

        public int NumHeads { get; set; } = 4;

        public int NumLayers { get; set; } = 4;

        public int FeedForwardDim { get; set; } = 256;

        /// <summary>True = causal (decoder-only) masking; false = bidirectional.</summary>
        public bool UseDecoderOnly { get; set; }

        public override void Validate()
        {
            RequirePositive(EmbeddingDim, nameof(EmbeddingDim));
            RequirePositive(NumHeads, nameof(NumHeads));
            RequirePositive(NumLayers, nameof(NumLayers));
            RequirePositive(FeedForwardDim, nameof(FeedForwardDim));
            RequireDivisible(EmbeddingDim, NumHeads, nameof(EmbeddingDim), nameof(NumHeads));
        }
    }

    /// <summary>
    /// Text encoder block configuration.
    /// Used by: CrossAttentionMultimodal, TACAMT, MMTAC.
    /// </summary>
    public sealed class TextEncoderConfig : TransformerModuleConfigBase
    {
        public int VocabSize { get; set; } = 50000;

        public int MaxSequenceLength { get; set; } = 1024;

        /// <summary>
        /// When true, text-encoder weights receive no gradient updates during training.
        /// </summary>
        public bool Freeze { get; set; } = false;

        public TextEncoderConfig() { UseDecoderOnly = false; }

        public override void Validate()
        {
            base.Validate();
            RequirePositive(VocabSize, nameof(VocabSize));
            RequirePositive(MaxSequenceLength, nameof(MaxSequenceLength));
        }
    }

    /// <summary>
    /// Price decoder block configuration.
    /// Used by: CrossAttentionMultimodal, TACAMT, MMTAC.
    /// </summary>
    public sealed class PriceDecoderConfig : TransformerModuleConfigBase
    {
        /// <summary>Number of input features per timestep (e.g. OHLCV = 5).</summary>
        public int InputFeatureDim { get; set; } = 5;

        public int MaxSequenceLength { get; set; } = 256;

        public PriceDecoderConfig() { UseDecoderOnly = true; }

        public override void Validate()
        {
            base.Validate();
            RequirePositive(InputFeatureDim, nameof(InputFeatureDim));
            RequirePositive(MaxSequenceLength, nameof(MaxSequenceLength));
        }
    }

    /// <summary>
    /// Output projection head configuration.
    /// Used by: CrossAttentionMultimodal, TACAMT, MultimodalTransformerConfig.
    /// </summary>
    public sealed class OutputHeadConfig : ConfigBase
    {
        /// <summary>Regression output count per timestep (e.g. 1 = close, 5 = OHLCV).</summary>
        public int OutputDim { get; set; } = 1;

        public bool UseConfidenceHead { get; set; } = false;

        public int EffectiveOutputDim => UseConfidenceHead ? OutputDim + 1 : OutputDim;

        public override void Validate()
        {
            RequirePositive(OutputDim, nameof(OutputDim));
        }
    }

    /// <summary>
    /// Hardware acceleration and FFN activation settings.
    /// Used by: all four architectures.
    /// </summary>
    public sealed class RuntimeConfig : ConfigBase
    {
        public ActivationType FFNActivationType { get; set; } = ActivationType.Relu;

        public AccelerationType AccelerationType { get; set; } = AccelerationType.MultiThreadCPU;

        public int AccelerationDeviceId { get; set; } = 0;

        public override void Validate()
        {
            if (!Enum.IsDefined(typeof(ActivationType), FFNActivationType))
                throw new ArgumentException($"Invalid {nameof(FFNActivationType)} value: {FFNActivationType}.");

            if (!Enum.IsDefined(typeof(AccelerationType), AccelerationType))
                throw new ArgumentException($"Invalid {nameof(AccelerationType)} value: {AccelerationType}.");

            if (AccelerationDeviceId < 0)
                throw new ArgumentException($"{nameof(AccelerationDeviceId)} must be non-negative, got {AccelerationDeviceId}.");
        }
    }

    /// <summary>
    /// L2 regularization and gradient-clipping thresholds.
    /// Used by: all four architectures.
    /// </summary>
    public sealed class RegularizationConfig : ConfigBase
    {
        public float L2RegulationLamda { get; set; } = 0.01f;

        public float GradientClippingThreshold { get; set; } = 0.5f;

        public override void Validate()
        {
            RequireNonNegative(L2RegulationLamda, nameof(L2RegulationLamda));
            RequirePositive(GradientClippingThreshold, nameof(GradientClippingThreshold));
        }
    }

    /// <summary>
    /// Content-aware decay network configuration.
    /// Used by: TACAMT, MMTAC (and optionally MultimodalTransformerConfig).
    /// Set Enabled = false to skip validation of all other fields.
    /// </summary>
    public sealed class DecayNetworkConfig : ConfigBase
    {
        public bool Enabled { get; set; } = false;

        public int ProjectionDim { get; set; } = 16;

        public int HiddenDim { get; set; } = 32;

        public int TimeEncodingBases { get; set; } = 16;

        public float MemAttentionDropout { get; set; } = 0.2f;

        public float MlpDropout { get; set; } = 0.2f;

        public float WeightDecay { get; set; } = 0.0f;

        public override void Validate()
        {
            if (!Enabled) return;

            RequirePositive(ProjectionDim, nameof(ProjectionDim));
            RequirePositive(HiddenDim, nameof(HiddenDim));
            RequirePositive(TimeEncodingBases, nameof(TimeEncodingBases));
            RequireInRangeZeroToOneExclusiveUpper(MemAttentionDropout, nameof(MemAttentionDropout));
            RequireInRangeZeroToOneExclusiveUpper(MlpDropout, nameof(MlpDropout));
            RequireNonNegative(WeightDecay, nameof(WeightDecay));
        }
    }

    /// <summary>
    /// Price-context training settings (history/current split).
    /// Used by: TACAMT, MMTAC (and optionally MultimodalTransformerConfig).
    /// Set Enabled = false to skip validation of other fields.
    /// </summary>
    public sealed class PriceContextConfig : ConfigBase
    {
        public bool Enabled { get; set; } = false;

        public int MinHistoryLength { get; set; } = 5;

        public int MinCurrentLength { get; set; } = 5;

        public override void Validate()
        {
            if (!Enabled) return;

            RequirePositive(MinHistoryLength, nameof(MinHistoryLength));
            RequirePositive(MinCurrentLength, nameof(MinCurrentLength));
        }
    }

    /// <summary>
    /// Attention-score-based memory pruning settings.
    /// Used by: TACAMT, MMTAC.
    /// Controls how stale memory entries are evicted from the news/price memory banks.
    /// </summary>
    public sealed class MemoryPruningConfig : ConfigBase
    {
        /// <summary>
        /// EMA decay factor for updating per-entry attention scores.
        /// score = alpha × new_score + (1 − alpha) × old_score.
        /// Lower = longer memory of past attention.
        /// </summary>
        public float AttentionScoreAlpha { get; set; } = 0.1f;

        /// <summary>
        /// Minimum query exposures before attention-based pruning is applied
        /// to an entry (cold-start protection).
        /// </summary>
        public int MinQueryCountForPruning { get; set; } = 3;

        /// <summary>
        /// Fraction of capacity always reserved for the most recent entries,
        /// regardless of attention score.
        /// </summary>
        public float NewEntryReserveFraction { get; set; } = 0.1f;

        /// <summary>True = attention-score eviction; false = oldest-first eviction.</summary>
        public bool UseAttentionBasedPruning { get; set; } = true;

        public override void Validate()
        {
            RequireInRangeZeroToOneExclusiveUpper(AttentionScoreAlpha, nameof(AttentionScoreAlpha));
            RequirePositive(MinQueryCountForPruning, nameof(MinQueryCountForPruning));
            RequireInRangeZeroToOneExclusiveUpper(NewEntryReserveFraction, nameof(NewEntryReserveFraction));
        }
    }
 
}