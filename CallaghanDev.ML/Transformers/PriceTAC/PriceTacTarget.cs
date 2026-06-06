using System;

namespace CallaghanDev.ML.Transformers.PriceTAC
{
    public sealed class PriceTacTarget
    {
        // Regression (linear, MSE)
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }

        // Separate constrained heads
        public float Range { get; set; }
        public float Quality { get; set; }

        // Classification heads
        public float Direction { get; set; }
        public float MidWindowDirection { get; set; }

        public float[] ToRegressionArray() => new[] { High, Low, Close };

        public static PriceTacTarget FromArrays(float[] regression, float range, float quality, float direction, float midDir)
        {
            if (regression == null || regression.Length < 3)
            {
                throw new ArgumentException("regression must have at least 3 elements (High, Low, Close).");
            }

            return new PriceTacTarget
            {
                High = regression[0],
                Low = regression[1],
                Close = regression[2],
                Range = range,
                Quality = quality,
                Direction = direction,
                MidWindowDirection = midDir
            };
        }
    }
}
