using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MMTAC
{
    public sealed class ModelPrediction
    {
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }
        public float Range { get; set; }
        public float Quality { get; set; }
        public float DirectionProb { get; set; }
        public float MidWindowDirectionProb { get; set; }
        public float Confidence { get; set; } = 1f;
    }
}
