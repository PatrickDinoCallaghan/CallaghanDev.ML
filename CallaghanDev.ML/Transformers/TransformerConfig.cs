using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using global::CallaghanDev.ML.AccelerationManagers;
using global::CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CallaghanDev.ML.Transformers
{
    public class TransformerConfig
    {
        public int VocabSize { get; set; } = 50000;
        public int MaxSequenceLength { get; set; } = 512;
        public int EmbeddingDim { get; set; } = 512;
        public int NumHeads { get; set; } = 8;
        public int NumLayers { get; set; } = 6;
        public int FeedForwardDim { get; set; } = 2048;
        public ActivationType FFNActivationType { get; set; } = ActivationType.Relu;
        public bool UseDecoderOnly { get; set; } = true;

        //integration with my existing FFNeural network
        public AccelerationType AccelerationType { get; set; } = AccelerationType.CPU;
        public int AccelerationDeviceId { get; set; } = 0;
        public CostFunctionType CostFunction { get; set; } = CostFunctionType.mse;
        public ActivationDistribution ActivationDistribution { get; set; } = ActivationDistribution.Normal;
        public float L2RegulationLamda { get; set; } = 0.01f;
        public float GradientClippingThreshold { get; set; } = 1.0f;
    }

}