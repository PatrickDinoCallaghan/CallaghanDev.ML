using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.MMTAC
{
    public sealed class ModelTarget
    {
        // Regression (linear, MSE)
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }

        // Separate constrained heads
        /// <summary>High - Low.  Must be >= 0.  Softplus head, MSE loss.</summary>
        public float Range { get; set; }
        /// <summary>Reliability score in [0,1].  Sigmoid head, MSE loss.</summary>
        public float Quality { get; set; }

        // Classification (sigmoid, BCE)
        public float Direction { get; set; }
        public float MidWindowDirection { get; set; }

        /// <summary>Returns [High, Low, Close] — the 3 linear regression targets only.</summary>
        public float[] ToRegressionArray() => new[] { High, Low, Close };

        public static ModelTarget FromArrays(float[] regression, float range, float quality, float direction, float midDir)
        {
            if (regression == null || regression.Length < 3)
                throw new ArgumentException("regression must have at least 3 elements (High, Low, Close).");
            return new ModelTarget
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
