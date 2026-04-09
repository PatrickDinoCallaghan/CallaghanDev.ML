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
    /// Data type, training objective, and weight-initialisation settings.
    /// Used exclusively by: MultiTypeTransformer (TransformerConfig / LanguageModel).
    /// </summary>
    public sealed class DataConfig : ConfigBase
    {
        public TransformerDataType DataType { get; set; } = TransformerDataType.Text;

        public CostFunctionType CostFunction { get; set; } = CostFunctionType.mse;

        public ActivationDistribution ActivationDistribution { get; set; } = ActivationDistribution.Normal;

        public bool UsesDiscreteTokens =>
            DataType == TransformerDataType.Text ||
            DataType == TransformerDataType.SymbolicSequence;

        public bool UsesClassificationOutput() =>
            DataType != TransformerDataType.TimeSeriesRegression;

        public override void Validate()
        {
            if (!Enum.IsDefined(typeof(TransformerDataType), DataType))
                throw new ArgumentException($"Invalid {nameof(DataType)}: {DataType}.");

            if (!Enum.IsDefined(typeof(CostFunctionType), CostFunction))
                throw new ArgumentException($"Invalid {nameof(CostFunction)}: {CostFunction}.");

            if (!Enum.IsDefined(typeof(ActivationDistribution), ActivationDistribution))
                throw new ArgumentException($"Invalid {nameof(ActivationDistribution)}: {ActivationDistribution}.");
        }
    }


}
