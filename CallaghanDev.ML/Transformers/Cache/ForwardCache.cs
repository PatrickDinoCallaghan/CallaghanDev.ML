using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.Cache
{
    public class ForwardCache
    {
        public List<float[,]> LayerInputs { get; set; }
        public float[,] FinalHiddenStates;

        public List<AttentionCache> AttentionCaches { get; set; }

        public List<LayerNormCache> LN1Caches { get; set; }
        public List<LayerNormCache> LN2Caches { get; set; }

        public List<float[,]> FFNOutputs { get; set; }

        public List<float[][]> FFNInputs { get; set; }

        public float[,] EmbeddedInput { get; set; }

        public int[] TokenIds { get; set; }

        public float[,] ContinuousInput { get; set; }

        public ForwardCache(int numLayers)
        {
            LayerInputs = new List<float[,]>();
            AttentionCaches = new List<AttentionCache>();
            LN1Caches = new List<LayerNormCache>();
            LN2Caches = new List<LayerNormCache>();
            FFNOutputs = new List<float[,]>();
            FFNInputs = new List<float[][]>();

            for (int i = 0; i < numLayers; i++)
            {
                AttentionCaches.Add(new AttentionCache());
                LN1Caches.Add(new LayerNormCache());
                LN2Caches.Add(new LayerNormCache());
            }
        }
    }
}
