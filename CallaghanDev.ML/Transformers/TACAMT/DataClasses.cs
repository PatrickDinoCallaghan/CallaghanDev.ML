using System;

namespace CallaghanDev.ML.Transformers.TACAMT
{
    public class NewsStory
    {
        public int[] TokenIds { get; set; }
        public float ArrivalTime { get; set; }
        public NewsStory() { }
        public NewsStory(int[] tokenIds, float arrivalTime) { TokenIds = tokenIds; ArrivalTime = arrivalTime; }
    }

    public class NewsMemoryEntry
    {
        public int[] TokenIds { get; set; }

        public float[] HiddenState { get; set; }

        public double AbsoluteTimestamp { get; set; }

        public float AttentionScore { get; set; } = 1.0f;

        public int QueryCount { get; set; } = 0;
    }

    public class PriceMemoryEntry
    {
        public float[] HiddenState { get; set; }
        public double AbsoluteTimestamp { get; set; }

        /// <summary>
        /// Running exponential moving average of attention weight this entry receives.
        /// See NewsMemoryEntry.AttentionScore for explanation.
        /// </summary>
        public float AttentionScore { get; set; } = 1.0f;

        /// <summary>
        /// Number of times this entry has been queried (used for EMA warmup).
        /// </summary>
        public int QueryCount { get; set; } = 0;
    }

}
