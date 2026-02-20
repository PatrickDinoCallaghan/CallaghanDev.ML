using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.CrossAttentionMultimodal
{
    public class MultimodalTrainingConfig
    {
        public float LearningRate { get; set; } = 0; // auto-calculated from Epochs
        //public int WarmUpEpochs { get; set; } = 5;
        public int BatchSize { get; set; } = 4;
        public int Epochs { get; set; } = 50;
        public bool UseGradientClipping { get; set; } = true;
        public float GradientClipThreshold { get; set; } = 1.0f;
        public bool UseLearningRateDecay { get; set; } = false;
        public float LearningRateDecay { get; set; } = 0.95f;
        public int ValidationInterval { get; set; } = 10;
        public bool Verbose { get; set; } = true;

        public float ConfidenceLossWeight { get; set; } = 0.1f;
    }
}
