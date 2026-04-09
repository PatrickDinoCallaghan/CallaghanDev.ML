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
            base.Validate();
            RequirePositive(VocabSize, nameof(VocabSize));
            RequirePositive(MaxSequenceLength, nameof(MaxSequenceLength));
            RequirePositive(InputFeatureDim, nameof(InputFeatureDim));
            RequirePositive(OutputDim, nameof(OutputDim));

            if (Data == null) throw new ArgumentNullException(nameof(Data));
            if (Runtime == null) throw new ArgumentNullException(nameof(Runtime));
            if (Regularization == null) throw new ArgumentNullException(nameof(Regularization));

            Data.Validate();
            Runtime.Validate();
            Regularization.Validate();
        }
    }
}
