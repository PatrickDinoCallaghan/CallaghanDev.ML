using CallaghanDev.ML.Transformers.Configuration;
using System;
using System.Collections.Generic;

namespace CallaghanDev.ML.Transformers.PriceTAC
{
    public sealed class PriceTacGradients
    {
        public float[,] PriceInputProjectionGrad { get; set; }
        public float[] PriceInputProjectionBiasGrad { get; set; }

        public float[,] GlobalFeatureProjectionGrad { get; set; }
        public float[] GlobalFeatureBiasGrad { get; set; }

        public List<PriceTacBlockGradients> PriceBlockGrads { get; set; }

        public float[,] RegressionProjectionGrad { get; set; }
        public float[] RegressionBiasGrad { get; set; }

        public float[,] RangeProjectionGrad { get; set; }
        public float[] RangeBiasGrad { get; set; }

        public float[,] QualityProjectionGrad { get; set; }
        public float[] QualityBiasGrad { get; set; }

        public float[,] DirectionProjectionGrad { get; set; }
        public float[] DirectionBiasGrad { get; set; }

        public float[,] MidDirectionProjectionGrad { get; set; }
        public float[] MidDirectionBiasGrad { get; set; }

        public float[,] ConfidenceProjectionGrad { get; set; }
        public float[] ConfidenceBiasGrad { get; set; }

        // Shape-compatible with MMTAC context helpers: row 1 price-memory, row 2 global.
        public float[,] ContextTypeEmbeddingGrad { get; set; }

        public PriceTacGradients(PriceTacConfig cfg)
        {
            int priceEd = cfg.Price.EmbeddingDim;

            PriceInputProjectionGrad = new float[priceEd, cfg.Price.InputFeatureDim];
            PriceInputProjectionBiasGrad = new float[priceEd];

            if (cfg.Global.GlobalFeatureDim > 0)
            {
                GlobalFeatureProjectionGrad = new float[priceEd, cfg.Global.GlobalFeatureDim];
                GlobalFeatureBiasGrad = new float[priceEd];
            }

            PriceBlockGrads = new List<PriceTacBlockGradients>();
            for (int i = 0; i < cfg.Price.NumLayers; i++)
            {
                PriceBlockGrads.Add(new PriceTacBlockGradients(priceEd, cfg.Price.NumHeads, cfg.Decay.ProjectionDim, cfg.Decay.HiddenDim, cfg.Decay.TimeEncodingBases));
            }

            int rDim = PriceTacOutputConfig.RegressionOutputCount;
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

            ContextTypeEmbeddingGrad = new float[PriceTacConfig.ContextTypeCount, priceEd];
        }

        public void Zero()
        {
            ZeroMatrix(PriceInputProjectionGrad);
            Array.Clear(PriceInputProjectionBiasGrad, 0, PriceInputProjectionBiasGrad.Length);

            if (GlobalFeatureProjectionGrad != null)
            {
                ZeroMatrix(GlobalFeatureProjectionGrad);
                Array.Clear(GlobalFeatureBiasGrad, 0, GlobalFeatureBiasGrad.Length);
            }

            foreach (var g in PriceBlockGrads) g.Zero();

            ZeroMatrix(RegressionProjectionGrad);
            Array.Clear(RegressionBiasGrad, 0, RegressionBiasGrad.Length);

            ZeroMatrix(RangeProjectionGrad);
            Array.Clear(RangeBiasGrad, 0, RangeBiasGrad.Length);

            ZeroMatrix(QualityProjectionGrad);
            Array.Clear(QualityBiasGrad, 0, QualityBiasGrad.Length);

            ZeroMatrix(DirectionProjectionGrad);
            Array.Clear(DirectionBiasGrad, 0, DirectionBiasGrad.Length);

            ZeroMatrix(MidDirectionProjectionGrad);
            Array.Clear(MidDirectionBiasGrad, 0, MidDirectionBiasGrad.Length);

            if (ConfidenceProjectionGrad != null)
            {
                ZeroMatrix(ConfidenceProjectionGrad);
                Array.Clear(ConfidenceBiasGrad, 0, ConfidenceBiasGrad.Length);
            }

            ZeroMatrix(ContextTypeEmbeddingGrad);
        }

        private static void ZeroMatrix(float[,] m)
        {
            if (m != null)
                Array.Clear(m, 0, m.Length);
        }
    }
}
