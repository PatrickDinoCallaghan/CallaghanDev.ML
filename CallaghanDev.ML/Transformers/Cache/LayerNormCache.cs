using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.Cache
{
    public class LayerNormCache
    {
        public float[,] Input { get; set; }
        public float[] Mean { get; set; }
        public float[] Variance { get; set; }
        public float[,] Normalized { get; set; }
    }
}
