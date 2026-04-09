using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using CallaghanDev.ML.Transformers.TACAMT;


namespace CallaghanDev.ML.Transformers.MMTAC
{
    public class MmtacBlockGradients
    {
        public AttentionGradients SelfAttnGrads { get; }
        public LayerNormGradients LNSelfGrads { get; }
        public AttentionGradients CrossAttnGrads { get; }
        public LayerNormGradients LNCrossGrads { get; }
        public ContentAwareDecayGradients DecayGrads { get; }
        public LayerNormGradients LNFFNGrads { get; }

        public MmtacBlockGradients(int embeddingDim, int numHeads, int decayProjectionDim, int decayHiddenDim, int decayTimeBases)
        {
            SelfAttnGrads = new AttentionGradients(embeddingDim);
            LNSelfGrads = new LayerNormGradients(embeddingDim);
            CrossAttnGrads = new AttentionGradients(embeddingDim);
            LNCrossGrads = new LayerNormGradients(embeddingDim);
            LNFFNGrads = new LayerNormGradients(embeddingDim);

            if (numHeads > 0)
            {
                // The mlp inside the content aware decay network needs to take a concatenated vector as input
                // The query, the key, the intereaction and two scalar time features to append to that 
                int mlpInputDim = decayProjectionDim * 3 + 2;
                int timeRawDim = 2 * decayTimeBases;

                DecayGrads = new ContentAwareDecayGradients(numHeads, embeddingDim, decayProjectionDim, decayHiddenDim, mlpInputDim, decayTimeBases, timeRawDim);
            }
        }

        public void Zero()
        {
            SelfAttnGrads.Zero(); 
            LNSelfGrads.Zero();
            CrossAttnGrads.Zero();
            LNCrossGrads.Zero();
            LNFFNGrads.Zero(); 
            DecayGrads?.Zero();
        }
    }
}
