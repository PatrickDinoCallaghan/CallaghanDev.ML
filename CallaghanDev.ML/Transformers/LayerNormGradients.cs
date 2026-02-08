using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers
{

    public class LayerNormGradients
    {
        public float[] GammaGrad { get; set; }
        public float[] BetaGrad { get; set; }

        public LayerNormGradients(int size)
        {
            GammaGrad = new float[size];
            BetaGrad = new float[size];
        }

        public void Zero()
        {
            Array.Clear(GammaGrad, 0, GammaGrad.Length);
            Array.Clear(BetaGrad, 0, BetaGrad.Length);
        }
    }

}
