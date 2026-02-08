using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.Cache
{
    public class AttentionCache
    {
        public float[,] Q { get; set; }
        public float[,] K { get; set; }
        public float[,] V { get; set; }
        public float[,] AttentionScores { get; set; }
        public float[,] AttentionWeights { get; set; }
        public float[,] AttentionOutput { get; set; }
        public float[,] Input { get; set; }
    }
}
