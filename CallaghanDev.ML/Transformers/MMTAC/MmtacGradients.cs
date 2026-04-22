using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MMTAC
{
    public class MmtacGradients
    {
        // Text encoder
        public float[,] TextEmbeddingGrad { get; set; }
        public List<AttentionGradients> TextAttnGrads { get; set; }
        public List<LayerNormGradients> TextLN1Grads { get; set; }
        public List<LayerNormGradients> TextLN2Grads { get; set; }

        // Price input projection
        public float[,] PriceInputProjectionGrad { get; set; }
        public float[] PriceInputProjectionBiasGrad { get; set; }

        // Global feature projection
        public float[,] GlobalFeatureProjectionGrad { get; set; }
        public float[] GlobalFeatureBiasGrad { get; set; }

        // Price decoder blocks
        public List<MmtacBlockGradients> PriceBlockGrads { get; set; }

        // Regression head: High, Low, Close (3 outputs, linear)
        public float[,] RegressionProjectionGrad { get; set; }
        public float[] RegressionBiasGrad { get; set; }

        // Range head (softplus)
        public float[,] RangeProjectionGrad { get; set; }
        public float[] RangeBiasGrad { get; set; }

        // Quality head (sigmoid)
        public float[,] QualityProjectionGrad { get; set; }
        public float[] QualityBiasGrad { get; set; }

        // Direction head (sigmoid BCE)
        public float[,] DirectionProjectionGrad { get; set; }
        public float[] DirectionBiasGrad { get; set; }

        // MidWindowDirection head (sigmoid BCE)
        public float[,] MidDirectionProjectionGrad { get; set; }
        public float[] MidDirectionBiasGrad { get; set; }

        // Confidence head (optional sigmoid BCE)
        public float[,] ConfidenceProjectionGrad { get; set; }
        public float[] ConfidenceBiasGrad { get; set; }

        // Context-type embedding: 3 rows - news(0), price(1), global(2)
        public float[,] ContextTypeEmbeddingGrad { get; set; }

        public MmtacGradients(MmtacConfig cfg)
        {
            int textEd = cfg.Text.EmbeddingDim;
            int priceEd = cfg.Price.EmbeddingDim;

            TextEmbeddingGrad = new float[cfg.Text.VocabSize, textEd];
            TextAttnGrads = new List<AttentionGradients>();
            TextLN1Grads = new List<LayerNormGradients>();
            TextLN2Grads = new List<LayerNormGradients>();

            for (int i = 0; i < cfg.Text.NumLayers; i++)
            {
                TextAttnGrads.Add(new AttentionGradients(textEd));
                TextLN1Grads.Add(new LayerNormGradients(textEd));
                TextLN2Grads.Add(new LayerNormGradients(textEd));
            }

            PriceInputProjectionGrad = new float[priceEd, cfg.Price.InputFeatureDim];
            PriceInputProjectionBiasGrad = new float[priceEd];

            if (cfg.Global.GlobalFeatureDim > 0)
            {
                GlobalFeatureProjectionGrad = new float[priceEd, cfg.Global.GlobalFeatureDim];
                GlobalFeatureBiasGrad = new float[priceEd];
            }

            PriceBlockGrads = new List<MmtacBlockGradients>();
            for (int i = 0; i < cfg.Price.NumLayers; i++)
            {
                PriceBlockGrads.Add(new MmtacBlockGradients(priceEd, cfg.Price.NumHeads, cfg.Decay.ProjectionDim, cfg.Decay.HiddenDim, cfg.Decay.TimeEncodingBases));
            }

            // Output head gradients
            int rDim = MmtacOutputConfig.RegressionOutputCount; // 3 high low close

            RegressionProjectionGrad = new float[rDim, priceEd];
            RegressionBiasGrad = new float[rDim];
            RangeProjectionGrad = new float[1, priceEd];
            RangeBiasGrad = new float[1];
            QualityProjectionGrad = new float[1, priceEd];
            QualityBiasGrad = new float[1];
            DirectionProjectionGrad = new float[1, priceEd];
            DirectionBiasGrad = new float[1];
            MidDirectionProjectionGrad = new float[1, priceEd];
            MidDirectionBiasGrad = new float[1];

            if (cfg.Output.UseConfidenceHead)
            {
                ConfidenceProjectionGrad = new float[1, priceEd];
                ConfidenceBiasGrad = new float[1];
            }

            ContextTypeEmbeddingGrad = new float[MmtacConfig.ContextTypeCount, priceEd];
        }

        public void Zero()
        {
            ZeroMatrix(TextEmbeddingGrad);
            foreach (var g in TextAttnGrads) g.Zero();
            foreach (var g in TextLN1Grads) g.Zero();
            foreach (var g in TextLN2Grads) g.Zero();

            ZeroMatrix(PriceInputProjectionGrad);
            Array.Clear(PriceInputProjectionBiasGrad, 0, PriceInputProjectionBiasGrad.Length);

            if (GlobalFeatureProjectionGrad != null)
            {
                ZeroMatrix(GlobalFeatureProjectionGrad);
                Array.Clear(GlobalFeatureBiasGrad, 0, GlobalFeatureBiasGrad.Length);
            }

            foreach (var g in PriceBlockGrads) g.Zero();

            ZeroMatrix(RegressionProjectionGrad); Array.Clear(RegressionBiasGrad, 0, RegressionBiasGrad.Length);
            ZeroMatrix(RangeProjectionGrad); Array.Clear(RangeBiasGrad, 0, RangeBiasGrad.Length);
            ZeroMatrix(QualityProjectionGrad); Array.Clear(QualityBiasGrad, 0, QualityBiasGrad.Length);
            ZeroMatrix(DirectionProjectionGrad); Array.Clear(DirectionBiasGrad, 0, DirectionBiasGrad.Length);
            ZeroMatrix(MidDirectionProjectionGrad); Array.Clear(MidDirectionBiasGrad, 0, MidDirectionBiasGrad.Length);

            if (ConfidenceProjectionGrad != null)
            {
                ZeroMatrix(ConfidenceProjectionGrad);
                Array.Clear(ConfidenceBiasGrad, 0, ConfidenceBiasGrad.Length);
            }

            ZeroMatrix(ContextTypeEmbeddingGrad);
        }

        private static void ZeroMatrix(float[,] m)
        {
            int r = m.GetLength(0), c = m.GetLength(1);

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    m[i, j] = 0f;
                }
            }
        }
    }
}
