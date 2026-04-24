using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MultiTypeTransformer
{
    /// <summary>
    /// Top-level configuration for MultiTypeTransformer / LanguageModel.
    /// Inherits block shape from <see cref="TransformerModuleConfigBase"/> and
    /// composes <see cref="DataConfig"/>, <see cref="RuntimeConfig"/>, and
    /// <see cref="RegularizationConfig"/> as sub-configs.
    /// </summary>
    public sealed class TransformerConfig : TransformerModuleConfigBase
    {
        //  Sequence / vocabulary 

        public int VocabSize { get; set; } = 50000;

        public int MaxSequenceLength { get; set; } = 512;

        public int InputFeatureDim { get; set; } = 1;

        public int OutputDim { get; set; } = 1;

        //  Derived 

        public int EffectiveOutputDim => Data.UsesDiscreteTokens ? VocabSize : OutputDim;

        //  Sub-configs 

        public DataConfig Data { get; set; } = new DataConfig();

        public RuntimeConfig Runtime { get; set; } = new RuntimeConfig
        {
            AccelerationType = AccelerationType.CPU,
            FFNActivationType = ActivationType.Relu,
            AccelerationDeviceId = 0,
        };

        public RegularizationConfig Regularization { get; set; } = new RegularizationConfig
        {
            L2RegulationLamda = 0.01f,
            GradientClippingThreshold = 1.0f,
        };

        //  Constructor defaults 

        public TransformerConfig()
        {
            EmbeddingDim = 512;
            NumHeads = 8;
            NumLayers = 6;
            FeedForwardDim = 2048;
            UseDecoderOnly = true;
        }

        //  Validation 
        public override void Validate()
        {
            if (Data == null) throw new ArgumentNullException(nameof(Data));
            if (Runtime == null) throw new ArgumentNullException(nameof(Runtime));
            if (Regularization == null) throw new ArgumentNullException(nameof(Regularization));

            base.Validate();

            RequirePositive(VocabSize, nameof(VocabSize));
            RequirePositive(MaxSequenceLength, nameof(MaxSequenceLength));
            RequirePositive(InputFeatureDim, nameof(InputFeatureDim));
            RequirePositive(OutputDim, nameof(OutputDim));

            Data.Validate();
            Runtime.Validate();
            Regularization.Validate();

            if (NumHeads <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumHeads), "NumHeads must be positive.");

            if (EmbeddingDim % NumHeads != 0)
                throw new ArgumentException($"EmbeddingDim ({EmbeddingDim}) must be divisible by NumHeads ({NumHeads}).", nameof(EmbeddingDim));

            int headDim = EmbeddingDim / NumHeads;
            if ((headDim & 1) != 0)
                throw new ArgumentException($"RoPE requires an even per-head dimension. EmbeddingDim / NumHeads = {headDim}.", nameof(EmbeddingDim));

            if (Data.UsesDiscreteTokens)
            {
                if (VocabSize < 2)
                    throw new ArgumentException("Discrete token models require VocabSize >= 2.", nameof(VocabSize));
            }
            else
            {
                if (InputFeatureDim < 1)
                    throw new ArgumentException("Continuous models require InputFeatureDim >= 1.", nameof(InputFeatureDim));

                if (Data.DataType == TransformerDataType.TimeSeriesClassification && OutputDim < 2)
                    throw new ArgumentException("TimeSeriesClassification requires OutputDim >= 2 classes.", nameof(OutputDim));
            }
        }

    }
}
