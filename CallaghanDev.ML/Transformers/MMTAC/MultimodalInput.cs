using CallaghanDev.ML.Transformers.TACAMT;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MMTAC
{
    public class MultimodalInput
    {
        public DateTime PredictionTimestamp { get; set; }
        public float[,] PriceSequence { get; set; }
        public float[] GlobalFeatures { get; set; }
        public NewsStory[] NewsStories { get; set; }
    }
}
