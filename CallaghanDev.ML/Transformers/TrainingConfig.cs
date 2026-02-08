using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CallaghanDev.ML.Transformers
{
    public class TrainingConfig
    {
        public float LearningRate { get; set; } = 0.0001f;
        public int BatchSize { get; set; } = 32;
        public int Epochs { get; set; } = 10;
        public float GradientClipThreshold { get; set; } = 1.0f;
        public bool UseGradientClipping { get; set; } = true;
        public int ValidationInterval { get; set; } = 100;
        public bool Verbose { get; set; } = true;

        public bool UseLearningRateDecay { get; set; } = false;
        public float LearningRateDecay { get; set; } = 0.95f;
    }
}